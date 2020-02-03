"""ResNet Two-Stream Model
Based on github.com/tensorflow/models/blob/master/official/resnet
"""

import tensorflow as tf
import resnet_building_blocks

SCOPE = "resnet_two_stream"
IMAGENET_SIZE = 224
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


class Model(object):
    """Base class for ResNet Two-Stream model."""


    def __init__(self, params):
        """Create a model for classifying a sequence of images."""
        self.block_sizes = params.resnet_block_sizes
        self.block_strides = params.resnet_block_strides
        self.conv_stride = params.resnet_conv_stride
        self.data_format = params.data_format
        self.dtype = params.dtype
        self.first_pool_size = params.resnet_first_pool_size
        self.first_pool_stride = params.resnet_first_pool_stride
        self.kernel_size = params.resnet_kernel_size
        self.num_classes = params.num_classes
        self.num_dense = params.num_dense
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
        """Add operations to learn features on frames and flows.

        Args:
            inputs: A tensor representing a batch of input image sequences.
            is_training: A boolean representing whether training is active.

        Returns:
            A tensor with shape [batch_size, num_classes]
        """
        with tf.variable_scope(scope, custom_getter=self._custom_dtype_getter):

            # Input for appearance is last frame in sequence
            frames = inputs[0] # [batch, seq, size, size, channels]
            appearance = frames[:,-1] # [batch, size, size, channels]
            # Resize frames to proper height and width
            # so we can warmstart from ImageNet pre-trained model
            appearance = tf.image.resize_images(appearance,
                [IMAGENET_SIZE, IMAGENET_SIZE])
            appearance = tf.cast(appearance, dtype=self.dtype)

            # Input for motion are stacked optical flows
            flows = inputs[1] # [batch, seq, size, size, 2]
            flows = tf.transpose(flows, perm=[0, 2, 3, 1, 4]) # [batch, size, size, seq, 2]
            # Stack the sequence of all x and then y flows as channels
            stack_flows = lambda x: tf.concat([x[...,0], x[...,1]], axis=2)
            motion = tf.map_fn(stack_flows, flows) # [batch, size, size, 32]
            # Resize flows to proper height and width
            # so we can warmstart from ImageNet pre-trained model
            motion = tf.image.resize_images(motion,
                [IMAGENET_SIZE, IMAGENET_SIZE])
            motion = tf.cast(motion, dtype=self.dtype)

            # Convert to channels_first if necessary (performance boost)
            if self.data_format == 'channels_first':
                appearance = tf.transpose(a=appearance, perm=[0, 3, 1, 2])
                motion = tf.transpose(a=motion, perm=[0, 3, 1, 2])

            # Add layer to get three input channels
            motion = tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=3,
                padding='same',
                data_format=self.data_format,
                activation=tf.nn.relu,
                name='conv2d_fix_channels')(motion)

            with tf.variable_scope("motion", custom_getter=self._custom_dtype_getter):

                # First conv layer
                appearance = resnet_building_blocks.conv2d_fixed_padding(
                    inputs=appearance, filters=self.num_filters,
                    kernel_size=self.kernel_size, strides=self.conv_stride,
                    data_format=self.data_format)
                appearance = tf.identity(appearance, 'initial_conv')

                # First pool layer
                appearance = tf.keras.layers.MaxPool2D(
                    pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='same',
                    data_format=self.data_format)(appearance)
                appearance = tf.identity(appearance, 'initial_max_pool')

                for i, num_blocks in enumerate(self.block_sizes):
                    num_filters = self.num_filters * (2**i)
                    appearance = resnet_building_blocks.conv2d_block_layer(
                        inputs=appearance, filters=num_filters, blocks=num_blocks,
                        strides=self.block_strides[i], is_training=is_training,
                        name='block_layer{}'.format(i+1),
                        data_format=self.data_format)

                appearance = resnet_building_blocks.batch_norm(
                    inputs=appearance, data_format=self.data_format)
                appearance = tf.nn.relu(appearance)

            with tf.variable_scope("appearance", custom_getter=self._custom_dtype_getter):

                # First conv layer
                motion = resnet_building_blocks.conv2d_fixed_padding(
                    inputs=motion, filters=self.num_filters,
                    kernel_size=self.kernel_size, strides=self.conv_stride,
                    data_format=self.data_format)
                motion = tf.identity(motion, 'initial_conv')

                # First pool layer
                motion = tf.keras.layers.MaxPool2D(
                    pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='same',
                    data_format=self.data_format)(motion)
                motion = tf.identity(motion, 'initial_max_pool')

                for i, num_blocks in enumerate(self.block_sizes):
                    num_filters = self.num_filters * (2**i)
                    motion = resnet_building_blocks.conv2d_block_layer(
                        inputs=motion, filters=num_filters, blocks=num_blocks,
                        strides=self.block_strides[i], is_training=is_training,
                        name='block_layer{}'.format(i+1),
                        data_format=self.data_format)

                motion = resnet_building_blocks.batch_norm(
                    inputs=motion, data_format=self.data_format)
                motion = tf.nn.relu(motion)

            # Fuse spatial features using concat + 2d conv
            axis = 1 if self.data_format == 'channels_first' else 3
            fused = tf.concat(values=[appearance, motion], axis=axis)
            fused = tf.keras.layers.Conv2D(
                filters=int(appearance.get_shape()[axis]),
                kernel_size=1,
                padding='same',
                data_format=self.data_format,
                activation=tf.nn.relu,
                name='conv2d_fusion')(fused)
            # Spatial average pooling
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            fused = tf.reduce_mean(input_tensor=fused, axis=axes, keepdims=True)
            fused = tf.identity(fused, 'final_reduce_mean')

            # Dense
            fused = tf.squeeze(fused, axes)
            logits = tf.keras.layers.Dense(units=self.num_classes)(fused)
            logits = tf.identity(logits, 'final_dense')

            return logits, fused
