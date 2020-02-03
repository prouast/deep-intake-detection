"""ResNet 2D CNN Model"""

import tensorflow as tf
import resnet_building_blocks

SCOPE = "resnet_2d_cnn"
IMAGENET_SIZE = 224
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for ResNet 2D CNN model."""


    def __init__(self, params):
        """Create a model for classifying an image."""
        self.block_sizes = params.resnet_block_sizes
        self.block_strides = params.resnet_block_strides
        self.conv_stride = params.resnet_conv_stride
        self.data_format = params.data_format
        self.dtype = params.dtype
        self.first_pool_size = params.resnet_first_pool_size
        self.first_pool_stride = params.resnet_first_pool_stride
        self.kernel_size = params.resnet_kernel_size
        self.num_classes = params.num_classes
        self.num_filters = params.resnet_num_filters


    def _custom_dtype_getter(self, getter, name, shape=None,
                             dtype=DEFAULT_DTYPE, *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary."""
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)


    def __call__(self, inputs, is_training, scope=SCOPE):
        """Add operations to learn features on a batch of images."""
        with tf.variable_scope(scope, custom_getter=self._custom_dtype_getter):

            # Determine the number of channels
            num_channels = inputs.get_shape()[3]

            # Resize input to proper height and width
            # so we can warmstart from ImageNet pre-trained model
            inputs = tf.image.resize_images(inputs,
                [IMAGENET_SIZE, IMAGENET_SIZE])
            inputs = tf.cast(inputs, dtype=self.dtype)

            # Convert to channels_first if necessary (performance boost)
            if self.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

            # If necessary, add layer to get three input channels
            if num_channels == 2:
                inputs = tf.keras.layers.Conv2D(
                    filters=3,
                    kernel_size=3,
                    padding='same',
                    data_format=self.data_format,
                    activation=tf.nn.relu,
                    name='conv2d_fix_channels')(inputs)

            # First conv layer
            inputs = resnet_building_blocks.conv2d_fixed_padding(
                inputs=inputs, filters=self.num_filters,
                kernel_size=self.kernel_size, strides=self.conv_stride,
                data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')

            # First pool layer
            inputs = tf.keras.layers.MaxPool2D(
                pool_size=self.first_pool_size, strides=self.first_pool_stride,
                padding='same', data_format=self.data_format)(inputs)
            inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2**i)
                inputs = resnet_building_blocks.conv2d_block_layer(
                    inputs=inputs, filters=num_filters, blocks=num_blocks,
                    strides=self.block_strides[i], is_training=is_training,
                    name='block_layer{}'.format(i+1),
                    data_format=self.data_format)

            inputs = resnet_building_blocks.batch_norm(
                inputs=inputs, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

            # Spatial average pooling
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            # Dense
            inputs = tf.squeeze(inputs, axes)
            logits = tf.keras.layers.Dense(units=self.num_classes)(inputs)
            logits = tf.identity(logits, 'final_dense')

            return logits, inputs
