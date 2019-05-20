import os
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf

tf.enable_eager_execution()

TFRECORDS_SUFFIX = '*.tfrecords'
CSV_SUFFIX = '.csv'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    name='input_dir', default='data', help='Directory for input data.')
tf.app.flags.DEFINE_integer(
    name='num_fc7', default=1024, help='Number of fc7 features.')

def get_file_id(filename):
    dir = os.path.dirname(filename)
    file_id = os.path.splitext(os.path.basename(filename))[0]
    return int(file_id)

def input_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example, {
            'example/video_id': tf.FixedLenFeature([], dtype=tf.int64),
            'example/seq_no': tf.FixedLenFeature([], dtype=tf.int64),
            'example/label': tf.FixedLenFeature([], dtype=tf.int64),
            'example/prob_1': tf.FixedLenFeature([], dtype=tf.float32),
            'example/fc7': tf.FixedLenFeature([FLAGS.num_fc7], dtype=tf.float32)
    })
    # Convert label to one-hot encoding
    seq_no = tf.cast(features['example/seq_no'], tf.int32)
    label = tf.cast(features['example/label'], tf.int32)
    prob = features['example/prob_1']

    return seq_no, label, prob

def fetch_data(filename):
    assert os.path.isfile(filename), "Couldn't find tfrecords file"
    dataset = tf.data.TFRecordDataset(filename)
    seq_nos = []; labels = []; probs = []
    for example in dataset:
        seq_no, label, prob = input_parser(example)
        seq_nos.append(seq_no.numpy())
        labels.append(label.numpy())
        probs.append(prob.numpy())
    return seq_nos, labels, probs

def main(unused_argv):
    """Main"""
    # Scan for tfrecords files
    filenames = gfile.Glob(os.path.join(FLAGS.input_dir, TFRECORDS_SUFFIX))
    if not filenames:
        raise RuntimeError('No tfrecords files found.')
    tf.logging.info("Found {0} tfrecords files.".format(str(len(filenames))))
    # Create a csv file for each tfrecords file
    for filename in filenames:
        file_id = get_file_id(filename)
        out_filename = os.path.join("", FLAGS.input_dir, str(file_id))
        if tf.gfile.Exists(out_filename):
            tf.logging.info("CSV file already exists. Skipping {0}.".format(filename))
            continue
        tf.logging.info("Working on {0}.".format(filename))
        # Fetch data of interest from tfrecords
        seq_nos, labels, probs = fetch_data(filename)
        tf.logging.info("Writing {0} examples to {1}.csv...".format(len(labels), out_filename))
        out_array = np.column_stack((np.full((len(labels)), file_id), seq_nos, labels, probs))
        np.savetxt("{0}.csv".format(out_filename), out_array, delimiter=",", fmt=['%i', '%i','%i','%f'])

if __name__ == '__main__':
    tf.app.run(main=main)
