"""Building blocks for oreba models."""

import tensorflow as tf


def channels_axis(inputs, params):
    """Return the axis index which houses the channels."""
    if params.data_format == 'channels_first':
        axis = 1
    else:
        axis = len(inputs.get_shape()) - 1
    return axis


def batch_norm(inputs, params, name):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=channels_axis(inputs, params), center=True, scale=True,
        name=name, fused=True)(inputs)


def conv2d_layers(inputs, is_training, params):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor (num_inputs, size, size, depth)
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        Conv features (num_inputs, num_dense).
    """
    convolved = inputs
    for i, num_filters in enumerate(params.oreba_num_conv):
        convolved_input = convolved
        # Add batch norm layer if enabled
        if params.batch_norm:
            convolved_input = batch_norm(
                inputs=convolved_input, params=params,
                name='norm_conv2d_%d' % i)
        # Add dropout layer if enabled and not first conv layer
        if i > 0 and params.dropout:
            convolved_input = tf.keras.layers.Dropout(
                rate=params.dropout,
                name='drop_conv2d_%d' % i)(convolved_input)
        # Add 2d convolution layer
        convolved = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=params.oreba_kernel_size,
            padding='same',
            data_format=params.data_format,
            activation=tf.nn.relu,
            name='conv2d_%d' % i)(convolved_input)
        # Add pooling layer
        convolved = tf.keras.layers.MaxPool2D(
            pool_size=params.oreba_pool_size,
            strides=params.oreba_pool_stride_size,
            data_format=params.data_format,
            name='pool_conv2d_%d' % i)(convolved)

    return convolved


def conv3d_layers(inputs, is_training, params):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor (num_inputs, seq_length, size, size, depth)
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        fc7 features. (num_inputs, num_dense)
    """
    convolved = inputs
    for i, num_filters in enumerate(params.oreba_num_conv):
        convolved_input = convolved
        # Add batch norm layer if enabled
        if params.batch_norm:
            convolved_input = batch_norm(
                inputs=convolved_input, params=params,
                name='norm_conv3d_%d' % i)
        # Add dropout layer if enabled and not first conv layer
        if i > 0 and params.dropout:
            convolved_input = tf.keras.layers.Dropout(
                rate=params.dropout,
                name='drop_conv3d_%d' % i)(convolved_input)
        # Add 3d convolution layer
        convolved = tf.keras.layers.Conv3D(
            filters=num_filters,
            kernel_size=params.oreba_kernel_size,
            padding='same',
            data_format=params.data_format,
            activation=tf.nn.relu,
            name='conv3d_%d' % i)(convolved_input)
        # Add pooling layer
        convolved = tf.keras.layers.MaxPool3D(
            pool_size=params.oreba_pool_size,
            strides=params.oreba_pool_stride_size,
            data_format=params.data_format,
            name='pool_conv3d_%d' % i)(convolved)

    return convolved


def dense_layer(inputs, is_training, params):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        Dense features.
    """
    # Add dropout layer if enabled
    if params.dropout:
        inputs = tf.keras.layers.Dropout(
            rate=params.dropout,
            name='drop_dense_0')(inputs)
    # Add dense layer
    inputs = tf.keras.layers.Dense(
        units=params.num_dense,
        activation=tf.nn.relu,
        name='dense_0')(inputs)
    return inputs


def lstm_layer(inputs, is_training, params):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor (batch_size, sequence_length, dense_features)
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        Features of the last layer.
            (batch_size, sequence_length, lstm_features)
    """
    inputs = tf.keras.layers.LSTM(
        units=params.oreba_num_lstm,
        return_sequences=True,
        name='lstm_0')(inputs)

    return inputs


def class_layer(inputs, is_training, params):
    """Return the output operation following the network architecture.
    Args:
        inputs: Input Tensor
        is_training: True if in training mode
        params: Hyperparameters
    Returns:
        Logits.
    """
    dense = inputs
    # Add dropout layer if enabled
    if params.dropout:
        dense = tf.keras.layers.Dropout(
            rate=params.dropout,
            name='drop_dense_1')(dense)
    # Add classification layer
    logits = tf.keras.layers.Dense(
        units=params.num_classes,
        name='dense_1')(inputs)

    return logits
