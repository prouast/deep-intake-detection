"""Runs a model on the OREBA dataset."""

import os
import math
import json
import tensorflow as tf
from tensorflow.python.platform import gfile
from absl import app
from absl import flags
from absl import logging
import run_loop
import oreba_2d_cnn
import resnet_2d_cnn
import oreba_3d_cnn
import resnet_3d_cnn
import oreba_cnn_lstm
import resnet_cnn_lstm
import oreba_two_stream
import resnet_two_stream
import oreba_slowfast
import resnet_slowfast

DATASET_NAME = "OREBA"
ORIGINAL_SIZE = 140
FRAME_SIZE = 128
NUM_CHANNELS = 3
SEQ_LENGTH = 1
NUM_SHARDS = 4
DTYPE_MAP = {"fp16": tf.float16, "fp32": tf.float32}
CATEGORY_MAP = {"main": 1, "sub": 2, "hand": 3, "utensil": 4}
NUM_CLASSES_MAP = {"main": 2, "sub": 3, "hand": 3, "utensil": 5}

FLAGS = flags.FLAGS
flags.DEFINE_float(name='base_learning_rate', default=3e-3,
    help='Base learning rate as input to Adam.')
flags.DEFINE_integer(name='batch_size', default=64,
    help='Batch size used for training.')
flags.DEFINE_enum(name='data_format', default="channels_last",
    enum_values=["channels_first", "channels_last"],
    help="Set the data format used in the model.")
flags.DEFINE_enum(name='dtype', default="fp32", enum_values=DTYPE_MAP.keys(),
    help='The TensorFlow datatype used for calculations {fp16, fp32}.')
flags.DEFINE_string(name='eval_dir', default='eval',
    help='Directory for eval data.')
flags.DEFINE_string(name='finetune_only', default='',
    help='What type of layers should be finetuned')
flags.DEFINE_enum(name='label_category', default="main",
    enum_values=CATEGORY_MAP.keys(),
    help='Label category for classification task {main, sub, hand, utensil}.')
flags.DEFINE_enum(name='mode', default="train_and_evaluate",
    enum_values=["train_and_evaluate", "predict_and_export_csv",
        "predict_and_export_tfrecord", "export_saved_model"],
    help='What mode should tensorflow be started in')
flags.DEFINE_string(name='model', default='oreba_2d_cnn',
    help='Select the model: {oreba_2d_cnn, oreba_3d_cnn, oreba_cnn_lstm, \
        oreba_two_stream, oreba_slowfast, resnet_2d_cnn, resnet_3d_cnn, \
        resnet_cnn_lstm, resnet_two_stream, resnet_slowfast}')
flags.DEFINE_string(name='model_dir', default='run',
    help='Output directory for model and training stats.')
flags.DEFINE_integer(name='num_frames', default=397890,
    help='Number of training images.')
flags.DEFINE_integer(name='num_parallel_calls', default=2,
    help='Number of parallel calls in input pipeline.')
flags.DEFINE_integer(name='num_sequences', default=396960,
    help='Number of training sequences.')
flags.DEFINE_integer(name='save_checkpoints_steps', default=100,
    help='Save checkpoints every x steps.')
flags.DEFINE_integer(name='save_summary_steps', default=25,
    help='Save summaries every x steps.')
flags.DEFINE_integer(name='sequence_shift', default=1,
    help='Shift taken in sequence generation.')
flags.DEFINE_integer(name='shuffle_buffer', default=50000,
    help='Buffer used for shuffling (~10x for img.')
flags.DEFINE_float(name='slowfast_alpha', default=4,
    help='Alpha parameter in SlowFast.')
flags.DEFINE_float(name='slowfast_beta', default=.25,
    help='Beta parameter in SlowFast.')
flags.DEFINE_string(name='train_dir', default='train',
    help='Directory for training data.')
flags.DEFINE_float(name='train_epochs', default=60,
    help='Number of training epochs.')
flags.DEFINE_boolean(name='use_distribution', default=False,
    help='Use tf.distribute.MirroredStrategy')
flags.DEFINE_boolean(name='use_flows', default=False,
    help='Use flows as features')
flags.DEFINE_boolean(name='use_frames', default=True,
    help='Use frames as features')
flags.DEFINE_boolean(name='use_sequence_input', default=False,
    help='Use a frame sequence instead of single frames')
flags.DEFINE_boolean(name='use_sequence_loss', default=False,
    help='Use sequence-to-sequence loss')
flags.DEFINE_string(name='warmstart_dir', default=None,
    help='Directory with checkpoints for warmstart')


def oreba_input_fn(is_training, use_sequence_input, use_frames, use_flows,
                   data_dir, label_category, dtype):
    """Input pipeline for Dataset API

    Args:
        is_training: Switch between training or eval input fn
        use_sequence_input: Enable/disable sliding window batch
        use_frames: Enable/disable frames as features
        use_flows: Enable/disable flows as features
        data_dir: The directory to the tfrecord files
        label_category: The label category to be used
        dtype: Data type for features/images

    Returns:
        The dataset ops
    """

    # Scan for training files
    filenames = gfile.Glob(os.path.join(data_dir, "*.tfrecord"))
    if not filenames:
        raise RuntimeError('No files found.')
    logging.info("Found {0} files.".format(str(len(filenames))))
    # List files
    files = tf.data.Dataset.list_files(filenames)
    # Shuffle files if needed
    if is_training:
        files = files.shuffle(NUM_SHARDS)

    # Read tfrecord, perform pre-processing and data augmentation
    dataset = files.interleave(
        lambda filename:
            tf.data.TFRecordDataset(filename)
            .map(map_func=_get_input_parser(use_frames, use_flows, label_category),
                num_parallel_calls=FLAGS.num_parallel_calls)
            .map(map_func=lambda id, no, i, f, l: (i, f, l))
            .apply(_get_sequence_batch_fn(use_sequence_input))
            .map(map_func=_get_transformation_parser(use_sequence_input, is_training,
                use_frames, use_flows), num_parallel_calls=FLAGS.num_parallel_calls)
            .map(map_func=_get_dtype_cast_parser(use_frames, use_flows, dtype),
                num_parallel_calls=FLAGS.num_parallel_calls)
            .map(map_func=lambda i, f, l: ((i, f), l),
                num_parallel_calls=FLAGS.num_parallel_calls),
        cycle_length=NUM_SHARDS)

    # Shuffle and repeat if needed
    if is_training:
        dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer).repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def _get_input_parser(use_frames, use_flows, label_category):
    """Return the input parser."""

    def input_parser(serialized_example):
        """Map serialized example to image data and label."""

        # Parse serialized example
        if use_flows:
            features = tf.io.parse_single_example(
                serialized_example, {
                    'example/video_id': tf.io.FixedLenFeature([], dtype=tf.string),
                    'example/seq_no': tf.io.FixedLenFeature([], dtype=tf.int64),
                    'example/label_{0}'.format(label_category): tf.io.FixedLenFeature([], dtype=tf.int64),
                    'example/image': tf.io.FixedLenFeature([], dtype=tf.string),
                    'example/flow': tf.io.FixedLenFeature([ORIGINAL_SIZE, ORIGINAL_SIZE, 2],
                        dtype=tf.float32)
            })
        else:
            features = tf.io.parse_single_example(
                serialized_example, {
                    'example/video_id': tf.io.FixedLenFeature([], dtype=tf.string),
                    'example/seq_no': tf.io.FixedLenFeature([], dtype=tf.int64),
                    'example/label_{0}'.format(label_category): tf.io.FixedLenFeature([], dtype=tf.int64),
                    'example/image': tf.io.FixedLenFeature([], dtype=tf.string)
            })

        # Video id
        id = features['example/video_id']

        # Sequence no
        seq_no = tf.cast(features['example/seq_no'], tf.int32)

        # Convert label to one-hot encoding
        label = tf.cast(features['example/label_{0}'.format(label_category)], tf.int32)

        if use_frames:
            # Convert data from scalar string tensor to uint8 tensor
            image_data = tf.decode_raw(features['example/image'], tf.uint8)
            image_data = tf.cast(image_data, tf.float32)
            # Reshape from [height * width * depth] to [height, width, depth].
            image_data = tf.reshape(image_data,
                [ORIGINAL_SIZE, ORIGINAL_SIZE, NUM_CHANNELS])
        else:
            image_data = []

        if use_flows:
            # Read flow data
            flow_data = features['example/flow']
        else:
            flow_data = []

        return id, seq_no, image_data, flow_data, label

    return input_parser


def _get_sequence_batch_fn(use_sequence_input):
    """Either return window-batched dataset, or identity"""
    if use_sequence_input:
        return lambda dataset: dataset.window(
            size=SEQ_LENGTH, shift=FLAGS.sequence_shift, drop_remainder=True).flat_map(
                lambda i, f, l: tf.data.Dataset.zip(
                    (i.batch(SEQ_LENGTH), f.batch(SEQ_LENGTH), l.batch(SEQ_LENGTH))))
    else:
        return lambda dataset: dataset


def _get_transformation_parser(use_sequence_input, is_training, use_frames, use_flows):
    """Return the data transformation parser."""

    def transformation_parser(image_data, flow_data, label_data):
        """Apply distortions to sequences or single images and flows."""

        if is_training:

            # Random rotation
            rotation_degree = tf.random.uniform([], -2.0, 2.0)
            rotation_radian = rotation_degree * math.pi / 180
            if use_frames:
                image_data = tf.contrib.image.rotate(image_data,
                    angles=rotation_radian)
            if use_flows:
                flow_data = tf.contrib.image.rotate(flow_data,
                    angles=rotation_radian)

            # Random crop
            diff = ORIGINAL_SIZE - FRAME_SIZE + 1
            limit = [1, diff, diff, 1] if use_sequence_input else [diff, diff, 1]
            offset = tf.random.uniform(shape=tf.shape(limit),
                dtype=tf.int32, maxval=tf.int32.max) % limit
            if use_frames:
                size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS] \
                    if use_sequence_input else [FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
                image_data = tf.slice(image_data, offset, size)
            if use_flows:
                size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, 2] \
                    if use_sequence_input else [FRAME_SIZE, FRAME_SIZE, 2]
                flow_data = tf.slice(flow_data, offset, size)

            # Random horizontal flip
            condition = tf.less(tf.random.uniform([], 0, 1.0), .5)
            if use_frames:
                image_data = tf.cond(pred=condition,
                    true_fn=lambda: tf.image.flip_left_right(image_data),
                    false_fn=lambda: image_data)
            if use_flows:
                flow_data = tf.cond(pred=condition,
                    true_fn=lambda: tf.image.flip_left_right(flow_data),
                    false_fn=lambda: flow_data)

            # Random brightness change
            def _adjust_brightness(image_data, delta):
                if tf.shape(image_data)[0] == 4:
                    brightness = lambda x: tf.image.adjust_brightness(x, delta)
                    return tf.map_fn(brightness, image_data)
                else:
                    return tf.image.adjust_brightness(image_data, delta)
            delta = tf.random.uniform([], -63, 63)
            if use_frames:
                image_data = _adjust_brightness(image_data, delta)

            # Random contrast change -
            def _adjust_contrast(image_data, contrast_factor):
                if tf.shape(image_data)[0] == 4:
                    contrast = lambda x: tf.image.adjust_contrast(x, contrast_factor)
                    return tf.map_fn(contrast, image_data)
                else:
                    return tf.image.adjust_contrast(image_data, contrast_factor)
            contrast_factor = tf.random.uniform([], 0.2, 1.8)
            if use_frames:
                image_data = _adjust_contrast(image_data, contrast_factor)

        else:

            # Crop the central [height, width].
            if use_frames:
                image_data = tf.image.resize_with_crop_or_pad(
                    image_data, FRAME_SIZE, FRAME_SIZE)
                size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS] \
                    if use_sequence_input else [FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS]
                image_data.set_shape(size)
            if use_flows:
                flow_data = tf.image.resize_with_crop_or_pad(
                    flow_data, FRAME_SIZE, FRAME_SIZE)
                size = [SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, 2] \
                    if use_sequence_input else [FRAME_SIZE, FRAME_SIZE, 2]
                flow_data.set_shape(size)

        # Subtract off the mean and divide by the variance of the pixels.
        def _standardization(images):
            """Linearly scales image data to have zero mean and unit variance."""
            num_pixels = tf.reduce_prod(tf.shape(images))
            images_mean = tf.reduce_mean(images)
            variance = tf.reduce_mean(tf.square(images)) - tf.square(images_mean)
            variance = tf.nn.relu(variance)
            stddev = tf.sqrt(variance)
            # Apply a minimum normalization that protects us against uniform images.
            min_stddev = tf.math.rsqrt(tf.cast(num_pixels, dtype=tf.float32))
            pixel_value_scale = tf.maximum(stddev, min_stddev)
            pixel_value_offset = images_mean
            images = tf.subtract(images, pixel_value_offset)
            images = tf.divide(images, pixel_value_scale)
            return images
        if use_frames:
            image_data = _standardization(image_data)

        return image_data, flow_data, label_data

    return transformation_parser


def _get_dtype_cast_parser(use_frames, use_flows, dtype):
    """Return the dtype cast parser."""

    def dtype_cast_parser(image_data, flow_data, label_data):
        """Cast dtype of sequences or single images and flows."""

        if use_frames:
            image_data = tf.cast(image_data, dtype=dtype)

        if use_flows:
            flow_data = tf.cast(flow_data, dtype=dtype)

        return image_data, flow_data, label_data

    return dtype_cast_parser


def get_label_category(label_category):
    return CATEGORY_MAP[label_category]


def get_num_classes(label_category):
    return NUM_CLASSES_MAP[label_category]


def get_tf_dtype(dtype):
    return DTYPE_MAP[dtype]


def oreba_model_fn(features, labels, mode, params):
    """Select the appropriate model_fn and model to run on OREBA."""

    model_params = tf.contrib.training.HParams(
        batch_norm=True,
        data_format=FLAGS.data_format,
        dropout=0.5,
        dtype=get_tf_dtype(FLAGS.dtype),
        frame_size=FRAME_SIZE,
        num_channels=NUM_CHANNELS,
        num_classes=get_num_classes(FLAGS.label_category),
        num_dense=1024,
        oreba_kernel_size=3,
        oreba_num_conv=[32, 32, 64, 64],
        oreba_num_lstm=128,
        oreba_pool_size=2,
        oreba_pool_stride_size=2,
        resnet_block_sizes=[3, 4, 6, 3],
        resnet_block_strides=[1, 2, 2, 2],
        resnet_conv_stride=2,
        resnet_first_pool_size=3,
        resnet_first_pool_stride=2,
        resnet_kernel_size=7,
        resnet_kernel_size_alt=5,
        resnet_num_filters=64,
        resnet_num_lstm=128,
        resnet_temporal_kernel_size_slow=1,
        resnet_temporal_kernel_size_fast=3,
        resnet_temporal_kernel_sizes_slow=[1, 1, 3, 3],
        resnet_temporal_kernel_sizes_fast=[3, 3, 3, 3],
        sequence_length=SEQ_LENGTH,
        slowfast_alpha=FLAGS.slowfast_alpha,
        slowfast_beta=FLAGS.slowfast_beta,
        use_sequence_loss=FLAGS.use_sequence_loss,
        warmstart=FLAGS.warmstart_dir is not None)

    # Set up features
    if FLAGS.mode == 'train_and_evaluate' or FLAGS.mode == 'predict_and_export_csv' or \
        FLAGS.mode == 'predict_and_export_tfrecord':

        if FLAGS.model == 'oreba_2d_cnn' or FLAGS.model == 'resnet_2d_cnn' or \
            FLAGS.model == 'oreba_cnn_lstm' or FLAGS.model == 'resnet_cnn_lstm' or \
            FLAGS.model == 'oreba_3d_cnn' or FLAGS.model == 'resnet_3d_cnn' or \
            FLAGS.model == 'oreba_slowfast' or FLAGS.model == 'resnet_slowfast':

            frames = features[0]; flows = features[1]
            if FLAGS.use_frames:
                assert not FLAGS.use_flows, "Cannot use frames with flows for this model."
                features = frames
            if FLAGS.use_flows:
                assert not FLAGS.use_frames, "Cannot use frames with flows for this model."
                features = flows

    elif FLAGS.mode == 'export_saved_model':

        assert not FLAGS.model == 'oreba_two_stream', "This model is not supported for export"
        assert not FLAGS.model == 'resnet_two_stream', "This model is not supported for export"
        features = features['frames']

    # Set up labels
    if FLAGS.use_sequence_input and mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.reshape(labels, [-1, params.sequence_length])

    # Set up model
    if FLAGS.model == 'oreba_2d_cnn':

        assert not FLAGS.use_sequence_input, "Cannot use sequence with this model."
        model = oreba_2d_cnn.Model(model_params)

    elif FLAGS.model == 'resnet_2d_cnn':

        assert not FLAGS.use_sequence_input, "Cannot use sequence with this model."
        model = resnet_2d_cnn.Model(model_params)

    elif FLAGS.model == 'oreba_3d_cnn':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.dtype == 'fp32', "fp16 not supported for 3d operations."
        model = oreba_3d_cnn.Model(model_params)

    elif FLAGS.model == 'resnet_3d_cnn':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.dtype == 'fp32', "fp16 not supported for 3d operations."
        # https://github.com/tensorflow/tensorflow/issues/7632
        # https://github.com/keras-team/keras/issues/9582 - Error appears for fp16 and seq
        model = resnet_3d_cnn.Model(model_params)

    elif FLAGS.model == 'oreba_cnn_lstm':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        model = oreba_cnn_lstm.Model(model_params)

    elif FLAGS.model == 'resnet_cnn_lstm':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        model = resnet_cnn_lstm.Model(model_params)

    elif FLAGS.model == 'oreba_two_stream':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.use_frames and FLAGS.use_flows, "Need both frames and flow for this model."
        model = oreba_two_stream.Model(model_params)

    elif FLAGS.model == 'resnet_two_stream':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.use_frames and FLAGS.use_flows, "Need both frames and flow for this model."
        model = resnet_two_stream.Model(model_params)

    elif FLAGS.model == 'oreba_slowfast':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.dtype == 'fp32', "fp16 not supported for 3d operations."
        model = oreba_slowfast.Model(model_params)

    elif FLAGS.model == 'resnet_slowfast':

        assert FLAGS.use_sequence_input, "Cannot use single images with this model."
        assert FLAGS.dtype == 'fp32', "fp16 not supported for 3d operations."
        model = resnet_slowfast.Model(model_params)

    else:
        raise RuntimeError('Invalid model selected.')

    model_fn = run_loop.model_fn(
        features=features,
        labels=labels,
        mode=mode,
        params=params,
        model=model)

    return model_fn


def _get_serving_input_receiver_fn():

    if FLAGS.use_sequence_input:
        features = {
            "frames": tf.compat.v1.placeholder(dtype=tf.float32,
                shape=[None, SEQ_LENGTH, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS])}
    else:
        features = {
            "frames": tf.compat.v1.placeholder(dtype=tf.float32,
                shape=[None, FRAME_SIZE, FRAME_SIZE, NUM_CHANNELS])}

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features=features,
        default_batch_size=None
    )

    return serving_input_receiver_fn


def oreba_warmstart_settings():
    """Determine the warmstart settings."""

    if FLAGS.warmstart_dir is None:
        return None

    # Using trained oreba_2d_cnn to warmstart oreba_cnn_lstm
    if FLAGS.warmstart_dir == 'oreba_2d_cnn' and FLAGS.model == 'oreba_cnn_lstm':
        # Warmstart all conv2d layers
        vars_to_warm_start = '(?!.*(lstm_0|dense_1))'
        var_name_to_prev_var_file = 'oreba_cnn_lstm_warmstart_from_oreba_2d_cnn.txt'

    # Using trained oreba_2d_cnn to warmstart oreba_two_stream
    elif FLAGS.warmstart_dir == 'oreba_2d_cnn' and FLAGS.model == 'oreba_two_stream':
        vars_to_warm_start = '(?!.*(conv2d_fix_channels|conv2d_fusion|dense_0|dense_1))'
        var_name_to_prev_var_file = 'oreba_two_stream_warmstart_from_oreba_2d_cnn.txt'

    # Using official resnet to warmstart resnet_2d_cnn
    elif FLAGS.warmstart_dir == 'resnet_fp16' and FLAGS.model == 'resnet_2d_cnn' or \
         FLAGS.warmstart_dir == 'resnet_fp32' and FLAGS.model == 'resnet_2d_cnn':
        # Warmstart all resnet_model layers except dense
        vars_to_warm_start = '(?!.*(dense|conv2d_fix_channels))'
        var_name_to_prev_var_file = 'resnet_2d_cnn_warmstart_from_resnet.txt'

    # Using resnet_2d_cnn to warmstart resnet_cnn_lstm
    elif FLAGS.warmstart_dir == 'resnet_2d_cnn' and FLAGS.model == 'resnet_cnn_lstm':
        # Warmstart all resnet_model layers except dense
        vars_to_warm_start = '(?!.*(dense|conv2d_fix_channels|lstm_0))'
        var_name_to_prev_var_file = 'resnet_cnn_lstm_warmstart_from_resnet_2d_cnn.txt'

    elif FLAGS.warmstart_dir == 'resnet_2d_cnn' and FLAGS.model == 'resnet_two_stream':
        # Warmstart all resnet_model layers except dense
        vars_to_warm_start = '(?!.*(dense|conv2d_fix_channels|conv2d_fusion))'
        var_name_to_prev_var_file = 'resnet_two_stream_warmstart_from_resnet_2d_cnn.txt'

    else:
        raise RuntimeError('Invalid warmstart and model settings.')

    # Load var_name_to_prev_var_name
    f = open(var_name_to_prev_var_file, 'r')
    var_name_to_prev_var_name = json.loads(f.read())
    if var_name_to_prev_var_name is None:
        raise RuntimeError('Could not locate warmstart variable map.')

    warmstart_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.warmstart_dir,
        vars_to_warm_start=vars_to_warm_start,
        var_name_to_prev_var_name=var_name_to_prev_var_name)

    return warmstart_settings


def run_oreba(flags_obj):
    """Run OREBA model training and eval loop."""
    flags = tf.contrib.training.HParams(
        base_learning_rate=FLAGS.base_learning_rate,
        batch_size=FLAGS.batch_size,
        dtype=get_tf_dtype(FLAGS.dtype),
        eval_dir=FLAGS.eval_dir,
        finetune_only=FLAGS.finetune_only,
        label_category=get_label_category(FLAGS.label_category),
        mode=FLAGS.mode,
        model_dir=FLAGS.model_dir,
        num_classes=get_num_classes(FLAGS.label_category),
        num_frames=FLAGS.num_frames,
        num_sequences=FLAGS.num_sequences,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_summary_steps=FLAGS.save_summary_steps,
        sequence_length=SEQ_LENGTH,
        sequence_shift=FLAGS.sequence_shift,
        train_dir=FLAGS.train_dir,
        train_epochs=FLAGS.train_epochs,
        use_distribution=FLAGS.use_distribution,
        use_flows=FLAGS.use_flows,
        use_frames=FLAGS.use_frames,
        use_sequence_input=FLAGS.use_sequence_input,
        use_sequence_loss=FLAGS.use_sequence_loss,
        warmstart_dir=FLAGS.warmstart_dir)

    run_loop.main(
        flags=flags,
        model_fn=oreba_model_fn,
        input_fn=oreba_input_fn,
        parse_fn=_get_input_parser(False, False, flags.label_category),
        serving_input_receiver_fn=_get_serving_input_receiver_fn(),
        warmstart_settings=oreba_warmstart_settings())


if __name__ == "__main__":
    app.run(
        main=run_oreba
    )
