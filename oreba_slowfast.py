"""Small SlowFast Model"""

import tensorflow as tf
import oreba_building_blocks

SCOPE = "oreba_slowfast"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for building small slowfast network."""


    def __init__(self, params):
        """Create a model to learn features on an object of the dimensions
            [seq_length, width, depth, channels].

        Args:
            params: Hyperparameters.
        """
        self.params = params
        self.dtype = params.dtype


    def _custom_dtype_getter(self, getter, name, shape=None,
                             dtype=DEFAULT_DTYPE, *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary."""
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)


    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to learn features on a batch of image sequences.

        Args:
            inputs: A tensor representing a batch of input image sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, num_classes]
        """
        with tf.compat.v1.variable_scope(scope, custom_getter=self._custom_dtype_getter):
            channels_first = self.params.data_format == 'channels_first'
            # Generate slow and fast inputs
            if self.params.slowfast_alpha == 8:
                slow = tf.stack([inputs[:,0], inputs[:,-1]], axis=1)
            elif self.params.slowfast_alpha == 4:
                slow = tf.stack([inputs[:,3], inputs[:,7], inputs[:,11], inputs[:,15]], axis=1)
            else:
                raise RuntimeError('Invalid slowfast_alpha selected.')
            fast = inputs
            # Convert to channels_first if necessary (GPU performance boost)
            if channels_first:
                slow = tf.transpose(a=slow, perm=[0, 4, 1, 2, 3])
                fast = tf.transpose(a=fast, perm=[0, 4, 1, 2, 3])
            # Blocks
            for i, num_filters in enumerate(self.params.oreba_num_conv):
                if self.params.batch_norm:
                    # Add batch norm layer if enabled
                    slow = oreba_building_blocks.batch_norm(
                        inputs=slow, params=self.params,
                        name='norm_slow_%d' % i)
                    fast = oreba_building_blocks.batch_norm(
                        inputs=fast, params=self.params,
                        name='norm_fast_%d' % i)
                # Add dropout layer if enabled and not first conv layer
                if i > 0 and self.params.dropout:
                    slow = tf.keras.layers.Dropout(
                        rate=self.params.dropout,
                        name='drop_slow_%d' % i)(slow)
                    fast = tf.keras.layers.Dropout(
                        rate=self.params.dropout,
                        name='drop_fast_%d' % i)(fast)
                # Add 3d convolution layers
                slow_kernel_size = self.params.oreba_kernel_size if i > 1 \
                    else [1, self.params.oreba_kernel_size, self.params.oreba_kernel_size]
                slow = tf.keras.layers.Conv3D(
                    filters=num_filters,
                    kernel_size=slow_kernel_size,
                    padding='same',
                    data_format=self.params.data_format,
                    activation=tf.nn.relu,
                    name='conv3d_slow_%d' % i)(slow)
                fast = tf.keras.layers.Conv3D(
                    filters=int(num_filters*self.params.slowfast_beta),
                    kernel_size=self.params.oreba_kernel_size,
                    padding='same',
                    data_format=self.params.data_format,
                    activation=tf.nn.relu,
                    name='conv3d_fast_%d' % i)(fast)
                # Add pooling layer - no pooling across temporal dimension
                slow = tf.keras.layers.MaxPool3D(
                    pool_size=[1, self.params.oreba_pool_size, self.params.oreba_pool_size],
                    strides=[1, self.params.oreba_pool_stride_size, self.params.oreba_pool_stride_size],
                    data_format=self.params.data_format,
                    name='pool_conv3d_slow_%d' % i)(slow)
                fast = tf.keras.layers.MaxPool3D(
                    pool_size=[1, self.params.oreba_pool_size, self.params.oreba_pool_size],
                    strides=[1, self.params.oreba_pool_stride_size, self.params.oreba_pool_stride_size],
                    data_format=self.params.data_format,
                    name='pool_conv3d_fast_%d' % i)(fast)
                # Lateral connection
                num_lateral_filters = int(2*self.params.slowfast_beta*num_filters)
                lateral_connection = tf.keras.layers.Conv3D(
                    filters=num_lateral_filters,
                    kernel_size=[self.params.oreba_kernel_size, 1, 1],
                    strides=[self.params.slowfast_alpha, 1, 1],
                    data_format=self.params.data_format,
                    padding='valid',
                    name='lateral_connection_%d' % i)(fast)
                # Fuse information into slow pathway by concatenation
                axis = 1 if channels_first else 4
                slow = tf.concat([slow, lateral_connection], axis=axis)

            # Temporal average pool
            axis = 2 if channels_first else 1
            slow = tf.reduce_mean(
                input_tensor=slow, axis=axis, keepdims=False)
            fast = tf.reduce_mean(
                input_tensor=fast, axis=axis, keepdims=False)
            axis = 1 if channels_first else 3
            fused = tf.concat([slow, fast], axis=axis)

            # 2D Conv fusion
            filters = self.params.oreba_num_conv[-1]
            fused = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                padding='same',
                data_format=self.params.data_format,
                activation=tf.nn.relu,
                name='conv2d_fusion')(fused)

            # Flatten
            fused = tf.keras.layers.Flatten()(fused)

            # Add dense layer
            fused = oreba_building_blocks.dense_layer(
                inputs=fused,
                is_training=is_training,
                params=self.params)

            # Add dense layer to compute logits
            logits = oreba_building_blocks.class_layer(
                inputs=fused,
                is_training=is_training,
                params=self.params)

        return logits, fused
