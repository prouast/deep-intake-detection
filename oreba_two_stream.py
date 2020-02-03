"""TwoStream Model"""

import tensorflow as tf
import oreba_building_blocks

SCOPE = "oreba_two_stream"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for building TwoStream network."""


    def __init__(self, params):
        """Create a model to learn features on an image and stacked optical
            flow. [2, batch_size, seq_length, width, depth, num_channels or 2]

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
        """Add operations to learn features on frames and flows.

        Args:
            inputs: A tensor representing a batch of input image sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, num_classes]
        """
        with tf.compat.v1.variable_scope(scope, custom_getter=self._custom_dtype_getter):
            # Input for appearance is last frame in sequence
            frames = inputs[0] # [batch, seq, size, size, channels]
            appearance = frames[:,-1] # [batch, size, size, channels]

            flows = inputs[1] # [batch, seq, size, size, 2]
            flows = tf.transpose(flows, perm=[0, 2, 3, 1, 4]) # [batch, size, size, seq, 2]
            stack_flows = lambda x: tf.concat([x[...,0], x[...,1]], axis=2)
            motion = tf.map_fn(stack_flows, flows) # [batch, size, size, 32]

            # Convert th channels_first if necessary (GPU performance boost)
            if self.params.data_format == 'channels_first':
                appearance = tf.transpose(a=appearance, perm=[0, 3, 1, 2])
                motion = tf.transpose(a=motion, perm=[0, 3, 1, 2])

            with tf.compat.v1.variable_scope("motion", custom_getter=self._custom_dtype_getter):
                if self.params.warmstart:
                    motion = tf.keras.layers.Conv2D(
                        filters=3,
                        kernel_size=self.params.oreba_kernel_size,
                        padding='same',
                        data_format=self.params.data_format,
                        activation=tf.nn.relu,
                        name='conv2d_fix_channels')(motion)
                motion = oreba_building_blocks.conv2d_layers(
                    inputs=motion,
                    is_training=is_training,
                    params=self.params)
            with tf.compat.v1.variable_scope("appearance", custom_getter=self._custom_dtype_getter):
                appearance = oreba_building_blocks.conv2d_layers(
                    inputs=appearance,
                    is_training=is_training,
                    params=self.params)
            # Fuse spatial features using concat + 2d conv
            axis = 1 if self.params.data_format == 'channels_first' else 3
            fused = tf.concat(values=[appearance, motion], axis=axis)
            fused = tf.keras.layers.Conv2D(
                filters=self.params.oreba_num_conv[-1],
                kernel_size=1,
                padding='same',
                data_format=self.params.data_format,
                activation=tf.nn.relu,
                name='conv2d_fusion')(fused)
            fused = tf.keras.layers.Flatten()(fused)
            fused = oreba_building_blocks.dense_layer(
                inputs=fused,
                is_training=is_training,
                params=self.params)
            logits = oreba_building_blocks.class_layer(
                inputs=fused,
                is_training=is_training,
                params=self.params)

        return logits, fused
