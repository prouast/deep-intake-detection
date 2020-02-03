"""ResNet-SlowFast Model
Based on github.com/tensorflow/models/blob/master/official/resnet
Adapted into SlowFast network as described by:
    SlowFast Networks for Video Recognition
    https://arxiv.org/pdf/1812.03982.pdf
    by Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He
Adapted for frame sequences of shape [16, 128, 128, 3]
"""

#from functools import reduce
import tensorflow as tf
import resnet_building_blocks

SCOPE = "resnet_slowfast"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for building the ResNet SlowFast model."""


    def __init__(self, params):
        """Create a model for classifying an image sequence."""
        self.block_sizes = params.resnet_block_sizes
        self.block_strides = params.resnet_block_strides
        self.data_format = params.data_format
        self.dtype = params.dtype
        self.first_pool_size = [1, params.resnet_first_pool_size, params.resnet_first_pool_size]
        self.first_pool_stride = [1, params.resnet_first_pool_stride, params.resnet_first_pool_stride]
        self.kernel_size_slow = [params.resnet_temporal_kernel_size_slow, params.resnet_kernel_size_alt, params.resnet_kernel_size_alt]
        self.kernel_size_fast = [params.resnet_temporal_kernel_size_fast, params.resnet_kernel_size_alt, params.resnet_kernel_size_alt]
        self.num_classes = params.num_classes
        self.num_filters = params.resnet_num_filters
        self.sequence_length = params.sequence_length
        self.slowfast_alpha = params.slowfast_alpha
        self.slowfast_beta = params.slowfast_beta
        self.temporal_kernel_sizes_slow = params.resnet_temporal_kernel_sizes_slow
        self.temporal_kernel_sizes_fast = params.resnet_temporal_kernel_sizes_fast
        self.use_sequence_loss = params.use_sequence_loss


    def _custom_dtype_getter(self, getter, name, shape=None,
                             dtype=DEFAULT_DTYPE, *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary."""
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)


    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to classify a batch of image sequences."""
        with tf.compat.v1.variable_scope(scope, custom_getter=self._custom_dtype_getter):
            channels_first = self.data_format == 'channels_first'
            if self.slowfast_alpha == 8:
                inputs_slow = tf.stack([inputs[:,0], inputs[:,-1]], axis=1)
            elif self.slowfast_alpha == 4:
                inputs_slow = tf.stack([inputs[:,3], inputs[:,7], inputs[:,11], inputs[:,15]], axis=1)
            else:
                raise RuntimeError('Invalid slowfast_alpha selected.')
            inputs_fast = inputs

            # Convert to channels_first if necessary (performance boost)
            if channels_first:
                inputs_slow = tf.transpose(a=inputs_slow, perm=[0, 4, 1, 2, 3])
                inputs_fast = tf.transpose(a=inputs_fast, perm=[0, 4, 1, 2, 3])

            # First conv layer
            inputs_slow = resnet_building_blocks.conv3d_fixed_padding(
                inputs=inputs_slow, filters=self.num_filters,
                kernel_size=self.kernel_size_slow, strides=1,
                data_format=self.data_format)
            inputs_slow = tf.identity(inputs_slow, 'initial_conv_slow')
            inputs_fast = resnet_building_blocks.conv3d_fixed_padding(
                inputs=inputs_fast, filters=int(self.num_filters * self.slowfast_beta),
                kernel_size=self.kernel_size_fast, strides=1,
                data_format=self.data_format)
            inputs_fast = tf.identity(inputs_fast, 'initial_conv_fast')

            # First pool layer
            inputs_slow = tf.keras.layers.MaxPool3D(
                pool_size=self.first_pool_size, strides=self.first_pool_stride,
                padding='same', data_format=self.data_format)(inputs_slow)
            inputs_slow = tf.identity(inputs_slow, 'initial_max_pool_slow')
            inputs_fast = tf.keras.layers.MaxPool3D(
                pool_size=self.first_pool_size, strides=self.first_pool_stride,
                padding='same', data_format=self.data_format)(inputs_fast)
            inputs_fast = tf.identity(inputs_fast, 'initial_max_pool_fast')

            # Lateral connection
            num_lateral_channels = int(2 * self.slowfast_beta * self.num_filters)
            lateral_connection = tf.keras.layers.Conv3D(
                filters=num_lateral_channels,
                kernel_size=[int(self.slowfast_alpha), 1, 1],
                strides=[int(self.slowfast_alpha), 1, 1], padding='valid',
                data_format=self.data_format,
                name='initial_lateral_connection')(inputs_fast)
            axis = 1 if channels_first else 4
            inputs_slow = tf.concat([inputs_slow, lateral_connection], axis=axis)

            # Block layers
            for i, num_blocks in enumerate(self.block_sizes):
                num_filters_slow = self.num_filters * (2**i)
                num_filters_fast = int(num_filters_slow * self.slowfast_beta)
                temporal_kernel_size_slow = self.temporal_kernel_sizes_slow[i]
                temporal_kernel_size_fast = self.temporal_kernel_sizes_fast[i]
                strides = [1, self.block_strides[i], self.block_strides[i]]
                inputs_slow = resnet_building_blocks.conv3d_block_layer(
                    inputs=inputs_slow, filters=num_filters_slow,
                    blocks=num_blocks, strides=strides,
                    temporal_kernel_size=temporal_kernel_size_slow,
                    is_training=is_training, name='block_layer_slow{}'.format(i+1),
                    data_format=self.data_format)
                inputs_fast = resnet_building_blocks.conv3d_block_layer(
                    inputs=inputs_fast, filters=num_filters_fast,
                    blocks=num_blocks, strides=strides,
                    temporal_kernel_size=temporal_kernel_size_fast,
                    is_training=is_training, name='block_layer_fast{}'.format(i+1),
                    data_format=self.data_format)

                # Lateral connection
                if i != 3:
                    num_lateral_channels = int(2 * self.slowfast_beta * 4 * num_filters_slow)
                    lateral_connection = tf.keras.layers.Conv3D(
                        filters=num_lateral_channels,
                        kernel_size=[int(self.slowfast_alpha), 1, 1],
                        strides=[int(self.slowfast_alpha), 1, 1], padding='valid',
                        data_format=self.data_format,
                        name='lateral_connection{}'.format(i+1))(inputs_fast)
                    axis = 1 if channels_first else 4
                    inputs_slow = tf.concat([inputs_slow, lateral_connection], axis=axis)

            inputs_slow = resnet_building_blocks.batch_norm(inputs_slow, self.data_format)
            inputs_slow = tf.nn.relu(inputs_slow)
            inputs_fast = resnet_building_blocks.batch_norm(inputs_fast, self.data_format)
            inputs_fast = tf.nn.relu(inputs_fast)

            # Average pooling
            axes = [2, 3, 4] if channels_first else [1, 2, 3]
            inputs_slow = tf.reduce_mean(
                input_tensor=inputs_slow, axis=axes, keepdims=False)
            inputs_slow = tf.identity(inputs_slow, 'pooled_slow')
            inputs_fast = tf.reduce_mean(
                input_tensor=inputs_fast, axis=axes, keepdims=False)
            inputs_fast = tf.identity(inputs_fast, 'pooled_fast')
            # Concat
            inputs = tf.concat([inputs_slow, inputs_fast], axis=1)

            # Final dense
            logits = tf.keras.layers.Dense(
                units=self.num_classes,
                name='final_dense')(inputs)

            return logits, inputs
