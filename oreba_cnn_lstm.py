"""CNN-LSTM Model"""

import tensorflow as tf
import oreba_building_blocks

SCOPE = "oreba_cnn_lstm"
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for building ConvLSTM network."""


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
            A tensor with shape [batch_size, seq_length, num_classes]
        """
        with tf.compat.v1.variable_scope(scope, custom_getter=self._custom_dtype_getter):
            # Reshape and feed all images through CNN
            num_channels = inputs.get_shape()[4]
            inputs = tf.reshape(inputs,
                [-1, self.params.frame_size, self.params.frame_size, num_channels])
            # Convert to channels_first if necessary (performance boost)
            if self.params.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
            inputs = oreba_building_blocks.conv2d_layers(
                inputs=inputs,
                is_training=is_training,
                params=self.params)
            inputs = tf.keras.layers.Flatten()(inputs)
            inputs = oreba_building_blocks.dense_layer(
                inputs=inputs,
                is_training=is_training,
                params=self.params)
            # Reshape and feed through sequence-aware LSTM
            inputs = tf.reshape(inputs,
                [-1, self.params.sequence_length, self.params.num_dense])
            inputs = oreba_building_blocks.lstm_layer(
                inputs=inputs,
                is_training=is_training,
                params=self.params)
            # Feed through classification layer
            logits = oreba_building_blocks.class_layer(
                inputs=inputs,
                is_training=is_training,
                params=self.params)

        return logits, inputs
