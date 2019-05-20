"""Run loop implemented for TF >= 1.9"""

from tensorflow.python.platform import gfile
import os
import math
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
import best_checkpoint_exporter

# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(flags, model_fn, input_fn, parse_fn, warmstart_settings):
    """Define and run the experiment.

    Args:
        flags: An object containing the flags from FLAGS.
        model_fn: The function that instantiates the model and builds the ops
            for train/eval.
        input_fn: The function that processes the dataset and returns a dataset
            that the estimator can train on.
        parse_fn: Dataset parsing function to be able to retrieve label
            information in predict mode.
        warmstart_settings: Settings for warmstarting the model from checkpoint
    """

    # Distribution strategy
    strategy = None
    if flags.use_distribution:
        strategy = tf.contrib.distribute.MirroredStrategy()

    # Run config
    run_config = tf.estimator.RunConfig(
        model_dir=flags.model_dir,
        save_summary_steps=flags.save_summary_steps,
        save_checkpoints_steps=flags.save_checkpoints_steps,
        train_distribute=strategy,
        eval_distribute=strategy)

    # Maximum steps based on the flags
    if flags.use_sequence_input:
        steps_per_epoch = int(flags.num_sequences / flags.batch_size \
                            * flags.sequence_shift / flags.sequence_length)
    else:
        steps_per_epoch = int(flags.num_frames / flags.batch_size)
    max_steps = steps_per_epoch * flags.train_epochs

    # Model parameters
    params = tf.contrib.training.HParams(
        base_learning_rate=flags.base_learning_rate,
        batch_size=flags.batch_size,
        decay_rate=0.9,
        dtype=flags.dtype,
        finetune_only=flags.finetune_only,
        gradient_clipping_norm=5.0,
        l2_lambda=1e-4,
        loss_scale=8.0,
        num_classes=flags.num_classes,
        sequence_length=flags.sequence_length,
        steps_per_epoch=steps_per_epoch,
        use_flows=flags.use_flows,
        use_frames=flags.use_frames,
        use_sequence_input=flags.use_sequence_input,
        use_sequence_loss=flags.use_sequence_loss)

    # Define the estimator.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=flags.model_dir,
        params=params,
        config=run_config,
        warm_start_from=warmstart_settings)

    # Exporters
    best_exporter = best_checkpoint_exporter.BestCheckpointExporter(
        score_metric='metrics/mean_per_class_accuracy',
        compare_fn=lambda x,y: x.score > y.score,
        sort_key_fn=lambda x: -x.score)

    # Basic profiling
    profiler_hook = tf.train.ProfilerHook(
        save_steps=flags.save_checkpoints_steps*100,
        output_dir=flags.model_dir,
        show_memory=True)

    # Training input_fn
    def train_input_fn():
        return input_fn(is_training=True, use_sequence_input=flags.use_sequence_input,
            use_frames=flags.use_frames, use_flows=flags.use_flows,
            data_dir=flags.train_dir, label_category=flags.label_category,
            dtype=flags.dtype)

    # Eval input_fn
    def eval_input_fn():
        return input_fn(is_training=False, use_sequence_input=flags.use_sequence_input,
            use_frames=flags.use_frames, use_flows=flags.use_flows,
            data_dir=flags.eval_dir, label_category=flags.label_category,
            dtype=flags.dtype)

    # Define the experiment
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=max_steps,
        hooks=[profiler_hook])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None, # Use evaluation feeder until it is empty
        exporters=best_exporter,
        start_delay_secs=120,
        throttle_secs=60)

    if flags.mode == "train_and_evaluate":
        # Start training and evaluation
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif flags.mode == "predict_and_export_csv":
        seq_skip = flags.sequence_length - 1 if flags.use_sequence_input else 0
        predict_and_export_csv(estimator, eval_input_fn, parse_fn,
            flags.eval_dir, seq_skip)
    elif flags.mode == "predict_and_export_tfrecords":
        seq_skip = flags.sequence_length - 1 if flags.use_sequence_input else 0
        predict_and_export_tfrecords(estimator, eval_input_fn, parse_fn,
            flags.eval_dir, seq_skip)


def model_fn(features, labels, mode, params, model):
    """Initializes a model for the estimator.

    Args:
        features: Input tensor representing a batch of input features
        labels: Labels tensor representing a batch of labels
        mode: Indicates which mode is running
        params: The parameters
        model: The model
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_predicting = mode == tf.estimator.ModeKeys.PREDICT

    # Generate a summary node for the images
    add_image_summaries(features, params)

    # Apply the model, if fp16 cast to fp32 for numerical stability.
    logits, fc7 = model(features, is_training)
    logits = tf.cast(logits, tf.float32)
    fc7 = tf.cast(fc7, tf.float32)

    # If necessary, slice last sequence step for logits
    final_logits = logits[:,-1,:] if logits.get_shape().ndims == 3 else logits
    final_fc7 = fc7[:,-1,:] if fc7.get_shape().ndims == 3 else fc7

    # Derive preds and probs from final_logits
    predictions = {
        'fc7': final_fc7,
        'classes': tf.argmax(final_logits, axis=-1),
        'probabilities': tf.nn.softmax(final_logits, name='softmax_tensor')}

    if is_predicting:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # If necessary, slice last sequence step for labels
    final_labels = labels[:,-1] if labels.get_shape().ndims == 2 else labels

    def _compute_balanced_sample_weight(labels):
        """Calculate the balanced sample weight for imbalanced data."""
        f_labels = tf.reshape(labels,[-1]) if labels.get_shape().ndims == 2 else labels
        y, idx, count = tf.unique_with_counts(f_labels)
        total_count = tf.size(f_labels)
        label_count = tf.size(y)
        calc_weight = lambda x: tf.divide(tf.divide(total_count, x),
            tf.cast(label_count, tf.float64))
        class_weights = tf.map_fn(fn=calc_weight, elems=count, dtype=tf.float64)
        sample_weights = tf.gather(class_weights, idx)
        sample_weights = tf.reshape(sample_weights, tf.shape(labels))
        return tf.cast(sample_weights, tf.float32)

    # Training with multiple labels per sequence
    if params.use_sequence_loss:

        if is_training:
            sample_weights = _compute_balanced_sample_weight(labels)
        else:
            sample_weights = tf.ones_like(labels, dtype=tf.float32)

        # Calculate and scale cross entropy
        scaled_loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=tf.cast(labels, tf.int32),
            weights=sample_weights)
        tf.identity(scaled_loss, name='seq2seq_loss')
        tf.summary.scalar('loss/seq2seq_loss', scaled_loss)

    # Training on single for frames or with one label per sequence
    else:

        if is_training:
            sample_weights = _compute_balanced_sample_weight(final_labels)
        else:
            sample_weights = tf.ones_like(final_labels, dtype=tf.float32)

        # Calculate scaled cross entropy
        unscaled_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(final_labels, tf.int32),
            logits=final_logits)
        scaled_loss = tf.reduce_mean(tf.multiply(unscaled_loss, sample_weights))
        tf.summary.scalar('loss/scaled_loss', scaled_loss)

    # Compute loss with Weight decay
    l2_loss = params.l2_lambda * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
            if 'norm' not in v.name])
    tf.summary.scalar('loss/l2_loss', l2_loss)
    loss = scaled_loss + l2_loss

    if is_training:
        global_step = tf.train.get_or_create_global_step()

        def _decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=params.steps_per_epoch, decay_rate=params.decay_rate)

        def _grad_filter(vars):
            """Only apply gradient updates to the certain layers."""
            if 'dense_lstm' in params.finetune_only:
                return [v for v in vars \
                    if 'dense' in v.name or 'lstm' in v.name]
            elif 'dense' in params.finetune_only:
                return [v for v in vars if 'dense' in v.name]
            else:
                return vars

        # Learning rate
        learning_rate = _decay_fn(params.base_learning_rate, global_step)
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('training/learning_rate', learning_rate)

        # The optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if params.dtype == tf.float16:
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(
                init_loss_scale=params.loss_scale, incr_every_n_steps=5)
            loss_scale_optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(
                optimizer, loss_scale_manager)
            grad_vars = loss_scale_optimizer.compute_gradients(loss)
        else:
            grad_vars = optimizer.compute_gradients(loss)

        # Filter vars to retain fine tuned vars
        grad_vars = _grad_filter(grad_vars)

        tf.summary.scalar("training/global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        # Clip gradients
        grads, vars = zip(*grad_vars)
        grads, _ = tf.clip_by_global_norm(grads, params.gradient_clipping_norm)
        grad_vars = list(zip(grads, vars))

        for grad, var in grad_vars:
            var_name = var.name.replace(":", "_")
            tf.summary.histogram("gradients/%s" % var_name, grad)
            tf.summary.scalar("gradient_norm/%s" % var_name, tf.global_norm([grad]))
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("training/clipped_global_gradient_norm",
            tf.global_norm(list(zip(*grad_vars))[0]))

        if params.dtype == tf.float16:
            minimize_op = loss_scale_optimizer.apply_gradients(grad_vars, global_step)
        else:
            minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

    else:
        train_op = None

    # Calculate accuracy metrics - always done with final labels
    final_labels = tf.cast(final_labels, tf.int64)
    accuracy = tf.metrics.accuracy(
        labels=final_labels, predictions=predictions['classes'])
    mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(
        labels=final_labels, predictions=predictions['classes'],
        num_classes=params.num_classes)
    tf.summary.scalar('metrics/accuracy', accuracy[1])
    tf.summary.scalar('metrics/mean_per_class_accuracy',
        tf.reduce_mean(mean_per_class_accuracy[1]))
    metrics = {
        'metrics/accuracy': accuracy,
        'metrics/mean_per_class_accuracy': mean_per_class_accuracy}

    # Calculate class-specific metrics
    for i in range(params.num_classes):
        class_precision = tf.metrics.precision_at_k(
            labels=final_labels, predictions=final_logits, k=1, class_id=i)
        class_recall = tf.metrics.recall_at_k(
            labels=final_labels, predictions=final_logits, k=1, class_id=i)
        tf.summary.scalar('metrics/class_%d_precision' % i, class_precision[1])
        tf.summary.scalar('metrics/class_%d_recall' % i, class_recall[1])
        metrics['metrics/class_%d_precision' % i] = class_precision
        metrics['metrics/class_%d_recall' % i] = class_recall

    # Log number of trainable model params
    trainable_params = [tf.reduce_prod(v.shape) for v in tf.trainable_variables()]
    tf.summary.scalar('model/trainable_params', sum(trainable_params))

    # Log number of total model params
    total_params = [tf.reduce_prod(v.shape) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    tf.summary.scalar('model/total_params', sum(total_params))

    # Log estimated checkpoint size
    weights_size = [tf.reduce_prod(v.shape) * v.dtype.size
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    tf.summary.scalar('model/weights_size_mb', sum(weights_size) / (1024 ** 2))

    # Log estimated back and forward activation size
    activations_size = [tf.reduce_prod(v.shape) * v.dtype.size * tf.constant(2) \
        * params.batch_size for v in tf.trainable_variables()]
    tf.summary.scalar('model/activations_size_mb', sum(activations_size) / (1024 ** 2))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def add_image_summaries(features, params):
    """Add sample summaries for train frames and flows."""

    # Set up frames and flows if available
    if params.use_flows and params.use_frames:
        frames = tf.cast(features[0], tf.float32)
        flows = tf.cast(features[1], tf.float32)
    elif params.use_frames:
        frames = tf.cast(features, tf.float32); flows = None
    elif params.use_flows:
        flows = tf.cast(features, tf.float32); frames = None

    def _flow_to_image(flow):
        """Convert flow representation for visualisation"""
        def _cart_to_polar(x, y):
            """Conversion from cart to polar coordinates."""
            theta = tf.atan2(y, x)
            rho = tf.sqrt(tf.add(tf.pow(x, 2), tf.pow(y, 2)))
            return theta, rho
        def _normalize(x, min, max):
            """Minmax normalization."""
            norm = tf.divide(
                tf.subtract(x, tf.reduce_min(x)),
                tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
            norm = tf.add(tf.multiply(norm, max-min), min)
            return norm
        ang, mag = _cart_to_polar(flow[:,:,0], flow[:,:,1])
        hsv_0 = tf.multiply(ang, tf.divide(180.0, tf.divide(math.pi, 2.0)))
        hsv_1 = tf.fill(tf.shape(hsv_0), 255.0)
        hsv_2 = _normalize(mag, 0, 255)
        res = tf.stack([hsv_0, hsv_1, hsv_2], axis=2)/255.0
        res = tf.image.hsv_to_rgb(res)
        return res

    if params.use_sequence_input:
        if frames is not None:
            for i in range(5):
                tf.summary.image(name="frame" + str(i),
                                tensor=frames[i],
                                max_outputs=params.sequence_length)
        if flows is not None:
            for i in range(5):
                temp = tf.map_fn(
                    fn=_flow_to_image,
                    elems=flows[i,0:params.sequence_length])
                tf.summary.image(name="flow" + str(i),
                                tensor=temp,
                                max_outputs=params.sequence_length)
    else:
        if frames is not None:
            tf.summary.image('frames', frames, max_outputs=10)
        if flows is not None:
            flows = tf.map_fn(_flow_to_image, flows[0:9])
            tf.summary.image('flows', flows, max_outputs=10)


def predict_and_export_csv(estimator, eval_input_fn, parse_fn, eval_dir, seq_skip):
    tf.logging.info("Working on {0}.".format(eval_dir))
    tf.logging.info("Starting prediction...")
    predictions = estimator.predict(input_fn=eval_input_fn)
    pred_list = list(itertools.islice(predictions, None))
    pred_probs_1 = list(map(lambda item: item["probabilities"][1], pred_list))
    num = len(pred_probs_1)
    # Get labels and ids
    filenames = gfile.Glob(os.path.join(eval_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(filenames))
    elem = dataset.map(parse_fn).make_one_shot_iterator().get_next()
    labels = []; id = []; seq_no = []; sess = tf.Session()
    for i in range(0, num + seq_skip):
        val = sess.run(elem)
        id.append(val[0])
        seq_no.append(val[1])
        labels.append(val[4])
    id = id[seq_skip:]; seq_no = seq_no[seq_skip:]; labels = labels[seq_skip:]
    assert (len(labels)==num), "Lengths must match"
    name = os.path.normpath(eval_dir).split(os.sep)[-1]
    tf.logging.info("Writing {0} examples to {1}.csv...".format(num, name))
    pred_array = np.column_stack((id, seq_no, labels, pred_probs_1))
    np.savetxt("{0}.csv".format(name), pred_array, delimiter=",", fmt=['%i','%i','%i','%f'])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.ravel()))


def predict_and_export_tfrecords(estimator, eval_input_fn, parse_fn, eval_dir, seq_skip):
    tf.logging.info("Working on {0}.".format(eval_dir))
    tf.logging.info("Starting prediction...")
    predictions = estimator.predict(input_fn=eval_input_fn)
    pred_list = list(itertools.islice(predictions, None))
    pred_fc7 = list(map(lambda item: item["fc7"], pred_list))
    pred_probs_1 = list(map(lambda item: item["probabilities"][1], pred_list))
    num = len(pred_fc7)
    tf.logging.info("Getting labels...")
    filenames = gfile.Glob(os.path.join(eval_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tf.data.Dataset.list_files(filenames))
    elem = dataset.map(parse_fn).make_one_shot_iterator().get_next()
    labels = []; id = []; seq_no = []; sess = tf.Session()
    for i in range(0, num + seq_skip):
        val = sess.run(elem)
        id.append(val[0])
        seq_no.append(val[1])
        labels.append(val[4])
    id = id[seq_skip:]; seq_no = seq_no[seq_skip:]; labels = labels[seq_skip:]
    name = os.path.normpath(eval_dir).split(os.sep)[-1]
    with tf.python_io.TFRecordWriter("{0}.tfrecords".format(name)) as tfrecord_writer:
        tf.logging.info("Writing {0} examples to {1}.tfrecords...".format(num, name))
        assert (len(labels)==num), "Lengths must match"
        pred_fc7 = np.array(pred_fc7)
        for index in range(num):
            example = tf.train.Example(features=tf.train.Features(feature={
                'example/video_id': _int64_feature(id[index]),
                'example/seq_no': _int64_feature(seq_no[index]),
                'example/label': _int64_feature(labels[index]),
                'example/prob_1': _floats_feature(pred_probs_1[index]),
                'example/fc7': _floats_feature(pred_fc7[index])
            }))
            tfrecord_writer.write(example.SerializeToString())
