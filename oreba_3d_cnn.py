"""3D CNN Model"""

import tensorflow as tf
import oreba_building_blocks

SCOPE = "oreba_3d_cnn"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for building 3d convolutional network."""


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
        with tf.variable_scope(scope, custom_getter=self._custom_dtype_getter):
            # Convert to channels_first if necessary (performance boost)
            if self.params.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 4, 1, 2, 3])
            inputs = oreba_building_blocks.conv3d_layers(
                inputs=inputs,
                is_training=is_training,
                params=self.params)
            inputs = tf.keras.layers.Flatten()(inputs)
            inputs = oreba_building_blocks.dense_layer(
                inputs=inputs,
                is_training=is_training,
                params=self.params)
            logits = oreba_building_blocks.class_layer(
                inputs=inputs,
                is_training=is_training,
                params=self.params)

        return logits, inputs
