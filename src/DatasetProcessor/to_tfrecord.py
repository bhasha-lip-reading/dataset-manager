import imageio
from tensorflow_docs.vis import embed
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import numpy as np

tfrecords_dir = "asset/tfrecords"
frames_dir = "asset/frames"
FRAME_COUNT = 35
WORDS_PER_BATCH = 25
IMG_SIZE = 48


if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder


def get_label(file):
    batch = int(file.split('-')[1])
    word = int(file.split('-')[-1])
    return (batch - 1) * WORDS_PER_BATCH + word


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(frames, label):
    feature = {}
    for i in range(FRAME_COUNT):
        feature['frame-%02d' % i] = image_feature(frames[i])
    feature['label'] = int64_feature(label)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_fn(example):
    feature_description = {}
    for i in range(FRAME_COUNT):
        feature_description['frame-%02d' %
                            i] = tf.io.FixedLenFeature([], tf.string)
    feature_description['label'] = tf.io.FixedLenFeature([], tf.int64)

    example = tf.io.parse_single_example(example, feature_description)

    frames = []
    for i in range(FRAME_COUNT):
        # example['frame-%02d' % i] = tf.io.decode_jpeg(example['frame-%02d' % i], channels=1)
        frames.append(tf.io.decode_jpeg(example['frame-%02d' % i], channels=1))
    return frames, example['label'] - 1


def to_tfrecords(speaker, video_files):
    # video_files = sorted(glob(frames_dir + '/*'))

    total_samples = len(video_files)
    num_samples = len(video_files)

    num_tfrecords = total_samples // num_samples
    if total_samples % num_samples:
        num_tfrecords += 1  # add one record if there are any remaining samples

    print("Number of tfrecords:", num_tfrecords)

    for tfrec_num in tqdm(range(num_tfrecords)):
        samples = video_files[(tfrec_num * num_samples)
                               : ((tfrec_num + 1) * num_samples)]

        with tf.io.TFRecordWriter(
            tfrecords_dir +
                "/speaker-{}-samples-{}.tfrecords".format(
                    speaker, len(video_files))
        ) as writer:
            for sample in samples:
                frames_path = sorted(glob(sample + '/*'))
                frames = []
                for frame_path in frames_path:
                    frame = tf.io.read_file(frame_path)
                    frame = tf.io.decode_jpeg(frame)
                    frames.append(frame)
                example = create_example(
                    frames, get_label(sample.split('/')[-1][:-3]))
                writer.write(example.SerializeToString())


def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=25)
    return embed.embed_file("animation.gif")


if __name__ == '__main__':
    
    for i in range(16):
        speaker = '{:03d}'.format(i + 1)
        video_files = sorted(glob(frames_dir + '/{}-*'.format(speaker)))
        # print(frames_dir + '/{}-*'.format(speaker))
        to_tfrecords(speaker, video_files)

    # sample = '/../003-03-03-03'
    # print(sample, sample.split('/')[-1][:-3],
    #       get_label(sample.split('/')[-1][:-3]))

    # from time import time

    # t = time()
    # """
    # read io"""
    # from glob import glob

    # files = sorted(glob('asset/tfrecords/speaker-{}*'.format(speaker)))

    # raw_dataset = tf.data.TFRecordDataset(files)
    # parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    # _ = None
    # __ = None

    # for video, label in parsed_dataset.take(1):
    #     _ = video
    #     __ = label

    # print(__.numpy())
    # print(_.numpy().shape)
    # to_gif(_.numpy())

    # print('Execution time: ', time() - t)
