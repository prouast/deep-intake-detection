"""Building blocks for resnet models.
Based on github.com/tensorflow/models/blob/master/official/resnet"""

import tensorflow as tf

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5


def channels_axis(inputs, data_format):
    """Return the axis index which houses the channels."""
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = len(inputs.get_shape()) - 1
    return axis


def batch_norm(inputs, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    return tf.keras.layers.BatchNormalization(
        axis=channels_axis(inputs, data_format), momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON, center=True, scale=True, fused=True,
        name='batch_normalization')(inputs)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, (seq,) channels, height_in, width_in] or
            [batch, (seq,) height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
            Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
    else:
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

    if len(inputs.get_shape()) == 5:
        paddings.insert(1, [0, 0])

    padded_inputs = tf.pad(tensor=inputs, paddings=paddings)

    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        data_format=data_format)(inputs)


def conv3d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 3-D convolution with explicit padding."""
    if (isinstance(strides, list) and max(strides) > 1) or \
       (isinstance(strides, int) and strides > 1):
        padding = 'valid'
        padding_kernel_size = max(kernel_size) if isinstance(kernel_size, list) else kernel_size
        inputs = fixed_padding(inputs, padding_kernel_size, data_format)
    else:
        padding = 'same'
    return tf.keras.layers.Conv3D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding=padding, use_bias=False,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        data_format=data_format)(inputs)


def conv2d_bottleneck_block_v2(inputs, filters, is_training, projection_shortcut,
                               strides, data_format):
    """A single block for ResNet v2 with bottleneck

    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of: Batch normalization then ReLU
    then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in]
            or [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for convolutions.
        is_training: Boolean indicating training or inference mode.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4*filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut


def conv3d_bottleneck_block_v2(inputs, filters, is_training, projection_shortcut,
                               strides, temporal_kernel_size, data_format):
    """A single block for ResNet v2 with bottleneck - adapted for 3D

    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Adapted to the ordering conventions of: Batch normalization then ReLU
    then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Adapted for frame sequences (own work)

    Args:
        inputs: A tensor of size [batch, seq, channels, height_in, width_in]
            or [batch, seq, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for convolutions.
        is_training: Boolean indicating training or inference mode.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride.
        temporal_kernel_size: Size of the temporal kernel in the first conv
            layer.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block; shape should match inputs.
    """
    shortcut = inputs
    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=[temporal_kernel_size, 1, 1],
        strides=1, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv3d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=[1, 3, 3],
        strides=strides, data_format=data_format)

    inputs = batch_norm(inputs, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv3d_fixed_padding(
        inputs=inputs, filters=4*filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut


def conv2d_block_layer(inputs, filters, blocks, strides, is_training, name,
                       data_format):
    """Create one layer of blocks for a 2D ResNet model.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        is_training: Are we currently training the model?
        name: A string name for the tensor output of the block layer.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4

    def _projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = conv2d_bottleneck_block_v2(
        inputs=inputs, filters=filters, is_training=is_training,
        projection_shortcut=_projection_shortcut, strides=strides,
        data_format=data_format)

    for _ in range(1, blocks):
        inputs = conv2d_bottleneck_block_v2(
            inputs=inputs, filters=filters, is_training=is_training,
            projection_shortcut=None, strides=1, data_format=data_format)

    return tf.identity(inputs, name)


def conv3d_block_layer(inputs, filters, blocks, strides, temporal_kernel_size,
                       is_training, name, data_format):
    """Create one layer of blocks for a 3D ResNet model.

    Args:
        inputs: A tensor of size [batch, seq, channels, height_in, width_in] or
            [batch, seq, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        temporal_kernel_size: Size of the temporal kernel in the first conv
            layer of each block.
        is_training: Are we currently training the model?
        name: A string name for the tensor output of the block layer.
        data_format: The input format ('channels_last' or 'channels_first').

    Returns:
        The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4

    def _projection_shortcut(inputs):
        return conv3d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1,
            strides=strides, data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = conv3d_bottleneck_block_v2(
        inputs=inputs, filters=filters, is_training=is_training,
        projection_shortcut=_projection_shortcut, strides=strides,
        temporal_kernel_size=temporal_kernel_size, data_format=data_format)

    for _ in range(1, blocks):
        inputs = conv3d_bottleneck_block_v2(
            inputs=inputs, filters=filters, is_training=is_training,
            projection_shortcut=None, strides=1,
            temporal_kernel_size=temporal_kernel_size, data_format=data_format)

    return tf.identity(inputs, name)
