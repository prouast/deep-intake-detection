"""Convert video files to tfrecords."""

import argparse
import datetime as dt
import os
import math
import logging
import sys
from collections import Counter
import csv
import cv2
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import xml.etree.cElementTree as etree

FLAGS = None
FILENAME_SUFFIX = 'tfrecords'
DEFAULT_LABEL = 'Idle'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _get_video_id(video_filename):
    dir = os.path.dirname(video_filename)
    video_id = os.path.splitext(os.path.basename(video_filename))[0]
    return int(video_id)


def _get_class_names():
    assert os.path.isfile(FLAGS.labels), "Couldn't find label master file"
    names_1 = []; names_2 = []; names_3 = []; names_4 = []
    tree = etree.parse(FLAGS.labels)
    categories = tree.getroot()
    for tag in categories[0]:
        names_1.append(tag.attrib['name'])
    for tag in categories[1]:
        names_2.append(tag.attrib['name'])
    for tag in categories[2]:
        names_3.append(tag.attrib['name'])
    for tag in categories[3]:
        names_4.append(tag.attrib['name'])
    return names_1, names_2, names_3, names_4


def _get_labels_filename(video_filename):
    dir = os.path.dirname(video_filename)
    file = os.path.splitext(os.path.basename(video_filename))[0]
    return os.path.join(dir, file + ".csv")


def _add_to_tfrecord(tfrecord_writer, video_id, frames, flows,
                     labels_1, labels_2, labels_3, labels_4, offset=0):
    """Add a batch of frames to tfrecords file

    Args:
        tfrecord_writer: The writer.
        video_id: The video id.
        frames: numpy array of frames to be written
        flows: numpy array of optical flow
        labels_1: numpy array of corresponding labels category 1
        labels_2: numpy array of corresponding labels category 2
        labels_3: numpy array of corresponding labels category 3
        labels_4: numpy array of corresponding labels category 4
        offset: Number of examples in tfrecords file thus far

    Returns:
        offset: Updated number of examples in tfrecords file
    """
    logging.info("Writing {0} examples with {1} labels for video {2}..."
        .format(frames.shape, labels_1.shape, video_id))

    # Assertions
    assert (frames.shape[0]==labels_1.shape[0]), \
        "Frame and label length must match!"
    assert (frames.shape[0]==flows.shape[0]), \
        "Frame and optical flow length must match!"

    # Grabbing the dimensions
    num = frames.shape[0]

    # Writing
    for index in range(num):
        image_raw = frames[index].tostring()
        flow_arr = flows[index].ravel()
        example = tf.train.Example(features=tf.train.Features(feature={
            'example/video_id': _int64_feature(video_id),
            'example/seq_no': _int64_feature(index),
            'example/label_1': _int64_feature(int(labels_1[index])),
            'example/label_2': _int64_feature(int(labels_2[index])),
            'example/label_3': _int64_feature(int(labels_3[index])),
            'example/label_4': _int64_feature(int(labels_4[index])),
            'example/image': _bytes_feature(image_raw),
            'example/flow': _floats_feature(flow_arr)
        }))
        tfrecord_writer.write(example.SerializeToString())

    return offset + num


def _video_to_numpy(filename):
    """Convert a video file to numpy array

    Args:
        filename: Filename of the video file

    Returns:
        timestamps: Numpy array of video frame timestamps
        frames: Numpy array of video frames
        num: Original number of frames
    """

    assert os.path.isfile(filename), "Couldn't find video file"

    cap = cv2.VideoCapture(filename)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    assert FLAGS.width == width, "Video width must be as specified in flag"
    assert FLAGS.height == height, "Video height must be as specified in flag"

    num = math.ceil(num_frames/FLAGS.keep_every)
    timestamps = np.empty(num, np.dtype('float32'))
    frames = np.empty((num, width, height, 3), np.dtype('uint8'))

    i = 0
    j = 0
    ret = True
    while (i < num_frames and ret):
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if i % FLAGS.keep_every == 0:
            timestamps[j] = timestamp
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[j] = frame
            j += 1
        i += 1

    cap.release()
    return timestamps, frames, num_frames


def _calc_opt_flow(frames):
    """Calculate optical flow as Dual TV-L1"""

    frame_1 = frames[0]
    prvs = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
    flows = np.empty((frames.shape[0]-1, frames.shape[1], frames.shape[2], 2), np.dtype('float32'))

    i = 0
    for frame_2 in frames[1:]:
        next = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
        dual_tv = cv2.optflow.createOptFlow_DualTVL1()
       	#dual_tv = cv2.DualTVL1OpticalFlow_create()
        flow = dual_tv.calc(prvs, next, None)
        flows[i] = flow
        i += 1

    return flows


def _timestamp_to_frame(time, timestamps):
    t = dt.datetime.strptime(time, '%M:%S.%f')
    ms = t.minute * 60 * 1000 + t.second * 1000 + t.microsecond/1000
    return np.argmax(timestamps >= ms)


def _label_to_numpy(filename, num, timestamps):

    assert os.path.isfile(filename), "Couldn't find label file"

    labels_1 = np.empty(num, dtype='U25'); labels_1.fill(DEFAULT_LABEL)
    labels_2 = np.empty(num, dtype='U25'); labels_2.fill(DEFAULT_LABEL)
    labels_3 = np.empty(num, dtype='U25'); labels_3.fill(DEFAULT_LABEL)
    labels_4 = np.empty(num, dtype='U25'); labels_4.fill(DEFAULT_LABEL)
    with open(filename) as dest_f:
        next(dest_f)
        for row in csv.reader(dest_f, delimiter=','):
            start_frame = _timestamp_to_frame(row[0], timestamps)
            end_frame = _timestamp_to_frame(row[1], timestamps)
            labels_1[start_frame:end_frame] = row[4]
            labels_2[start_frame:end_frame] = row[5]
            labels_3[start_frame:end_frame] = row[6]
            labels_4[start_frame:end_frame] = row[7]
    return labels_1, labels_2, labels_3, labels_4


def _label_to_int(label, class_names):
    return class_names.index(label)


def _add_to_class_counts(class_counts, labels):
    unique, counts = np.unique(labels, return_counts=True)
    new_class_counts = Counter(dict(zip(unique, counts)))
    return class_counts + new_class_counts


def _write_label_file(names_1, names_2, names_3, names_4,
                      counts_1, counts_2, counts_3, counts_4,
                      directory, filename):
    """Writes a file with the list of class names.
    Args:
        class_names: List of the class names.
        class_counts: Counter of class counts.
        directory: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(directory, filename)
    categories = etree.Element("categories")
    category_1 = etree.SubElement(categories, "category", id="1", name="Default")
    for name in names_1:
        id = _label_to_int(name, names_1)
        count = counts_1[name]
        etree.SubElement(category_1, "tag", id=str(id), name=name, count=str(count))
    category_2 = etree.SubElement(categories, "category", id="2", name="Intake")
    for name in names_2:
        id = _label_to_int(name, names_2)
        count = counts_2[name]
        etree.SubElement(category_2, "tag", id=str(id), name=name, count=str(count))
    category_3 = etree.SubElement(categories, "category", id="3", name="Hand")
    for name in names_3:
        id = _label_to_int(name, names_3)
        count = counts_3[name]
        etree.SubElement(category_3, "tag", id=str(id), name=name, count=str(count))
    category_4 = etree.SubElement(categories, "category", id="4", name="Utensil")
    for name in names_4:
        id = _label_to_int(name, names_4)
        count = counts_4[name]
        etree.SubElement(category_4, "tag", id=str(id), name=name, count=str(count))

    tree = etree.ElementTree(categories)
    tree.write(labels_filename)


def main(unused_argv):
    """Main"""

    # Set number of opencv threads to sequential to run on HPC
    cv2.setNumThreads(0)

    # Scan for video files
    filenames = gfile.Glob(os.path.join(FLAGS.directory, FLAGS.suffix))
    if not filenames:
        raise RuntimeError('No video files found.')
    logging.info("Found {0} video files.".format(str(len(filenames))))

    # Keep track of classes
    names_1, names_2, names_3, names_4 = _get_class_names()
    counts_1 = Counter(); counts_2 = Counter(); counts_3 = Counter(); counts_4 = Counter()

    # Create a separate TFRecords file for each video
    for filename in filenames:
        video_id = _get_video_id(filename)
        out_filename = os.path.join("", FLAGS.directory, str(video_id) + "." + FILENAME_SUFFIX)
        if tf.gfile.Exists(out_filename):
            logging.info("Dataset file already exists. Skipping {0}.".format(filename))
            continue
        with tf.python_io.TFRecordWriter(out_filename) as tfrecord_writer:
            logging.info("Working on {0}.".format(filename))
            # Fetch frames
            timestamps, frames, num_orig = _video_to_numpy(filename)
            # Compute optical flow and remove first frame
            flows = _calc_opt_flow(frames)
            frames = frames[1:]
            timestamps = timestamps[1:]
            # Fetch label filename and test for existence
            labels_filename = _get_labels_filename(filename)
            # Fetch labels
            labels_1, labels_2, labels_3, labels_4 = _label_to_numpy(
                labels_filename, len(frames), timestamps)
            # Update class names
            counts_1 = _add_to_class_counts(counts_1, labels_1)
            counts_2 = _add_to_class_counts(counts_2, labels_2)
            counts_3 = _add_to_class_counts(counts_3, labels_3)
            counts_4 = _add_to_class_counts(counts_4, labels_4)
            # Convert labels to int
            labels_1 = np.array([_label_to_int(i, names_1) for i in labels_1])
            labels_2 = np.array([_label_to_int(i, names_2) for i in labels_2])
            labels_3 = np.array([_label_to_int(i, names_3) for i in labels_3])
            labels_4 = np.array([_label_to_int(i, names_4) for i in labels_4])
            # Write to .tfrecords file
            num = _add_to_tfrecord(tfrecord_writer, video_id, frames, flows,
                                   labels_1, labels_2, labels_3, labels_4)
            logging.info("Finished writing {0} examples from {1} original " \
                "frames.".format(num, num_orig))

    logging.info("Final class counts for category 1: {0}.".format(counts_1))
    logging.info("Final class counts for category 2: {0}.".format(counts_2))
    logging.info("Final class counts for category 3: {0}.".format(counts_3))
    logging.info("Final class counts for category 4: {0}.".format(counts_4))

    # Write a labels file
    #labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    _write_label_file(names_1, names_2, names_3, names_4,
                      counts_1, counts_2, counts_3, counts_4,
                      FLAGS.directory, FLAGS.labels)

    logging.info("Finished converting the dataset!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='train',
        help='Directory to search for videos and labels'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='*.mp4',
        help='Suffix of video files'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='labels.xml',
        help='Filename for training labels'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=140,
        help='Height of video frames'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=140,
        help='Width of video frames'
    )
    parser.add_argument(
        '--keep_every',
        type=int,
        default=3,
        help='Keep every x frame in the video, resulting in framerate fps/4'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
