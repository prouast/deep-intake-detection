"""ResNet 3D CNN Model
Based on github.com/tensorflow/models/blob/master/official/resnet
Adapted for frame sequences of shape [16, 128, 128, 3]
"""

import tensorflow as tf
import resnet_building_blocks

SCOPE = "resnet_3d_cnn"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for ResNet 3D CNN model."""


    def __init__(self, params):
        """Create a model for classifying an image sequence."""
        self.block_sizes = params.resnet_block_sizes
        self.block_strides = params.resnet_block_strides
        self.data_format = params.data_format
        self.dtype = params.dtype
        self.first_pool_size = params.resnet_first_pool_size
        self.first_pool_stride = params.resnet_first_pool_stride
        self.kernel_size = [params.resnet_temporal_kernel_size_fast, params.resnet_kernel_size_alt, params.resnet_kernel_size_alt]
        self.num_classes = params.num_classes
        self.num_filters = params.resnet_num_filters
        self.temporal_kernel_sizes = params.resnet_temporal_kernel_sizes_fast


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

            # Convert to channels_first if necessary (performance boost)
            if self.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 4, 1, 2, 3])

            # First conv layer
            inputs = resnet_building_blocks.conv3d_fixed_padding(
                inputs=inputs, filters=self.num_filters,
                kernel_size=self.kernel_size, strides=1,
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            # First pool layer
            inputs = tf.keras.layers.MaxPool3D(
                pool_size=self.first_pool_size, strides=self.first_pool_stride,
                padding='same', data_format=self.data_format)(inputs)
            inputs = tf.identity(inputs, 'initial_max_pool')

            # Block layers
            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                temporal_kernel_size = self.temporal_kernel_sizes[i]
                inputs = resnet_building_blocks.conv3d_block_layer(
                    inputs=inputs, filters=num_filters,
                    blocks=num_blocks, strides=self.block_strides[i],
                    temporal_kernel_size=temporal_kernel_size,
                    is_training=is_training, name='block_layer{}'.format(i+1),
                    data_format=self.data_format)

            inputs = resnet_building_blocks.batch_norm(inputs, self.data_format)
            inputs = tf.nn.relu(inputs)

            # Spatial average pooling
            axes = [3, 4] if self.data_format == 'channels_first' else [2, 3]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=False)
            inputs = tf.identity(inputs, 'spatial_pooled')

            # Flatten
            inputs = tf.keras.layers.Flatten()(inputs)

            # Dense
            logits = tf.keras.layers.Dense(
                units=self.num_classes,
                name='final_dense')(inputs)

            return logits, inputs
