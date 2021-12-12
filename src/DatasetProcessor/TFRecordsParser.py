from abc import abstractproperty
import os
import unittest
import tensorflow as tf
import numpy as np
from glob import glob
from tensorflow.python.platform import gfile
from tqdm import tqdm
from tensorflow_docs.vis import embed
import imageio


height = 48
width = 48
num_depth = 3
in_path = "data/videos/"
out_path = "records/"
n_videos_per_record = 1


# class Testvideo2tfrecord(unittest.TestCase):
#     def test_example1(self):
#         n_frames = 5
#         convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
#                                    n_videos_in_record=n_videos_per_record,
#                                    n_frames_per_video=n_frames,
#                                    dense_optical_flow=True,
#                                    file_suffix="*.mp4")

#         filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
#         n_files = len(filenames)

#         self.assertTrue(filenames)
#         self.assertEqual(n_files * n_videos_per_record,
#                          get_number_of_records(filenames, n_frames))

#     " travis ressource exhaust, passes locally for 3.6 and 3.4"
# def test_example2(self):
#   n_frames = 'all'
#   convert_videos_to_tfrecord(source_path=in_path, destination_path=out_path,
#                              n_videos_in_record=n_videos_per_record,
#                              n_frames_per_video=n_frames,
#                              n_channels=num_depth, dense_optical_flow=False,
#                              file_suffix="*.mp4")
#
#   filenames = gfile.Glob(os.path.join(out_path, "*.tfrecords"))
#   n_files = len(filenames)
#
#   self.assertTrue(filenames)
#   self.assertEqual(n_files * n_videos_per_record,
#                    get_number_of_records(filenames, n_frames))


def read_and_decode(filename_queue, n_frames):
    """Creates one image sequence"""

    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq = []

    if n_frames == 'all':
        n_frames = 354  # travis kills due to too large tfrecord

    # label = None
    for image_count in range(n_frames):
        path = 'blob' + '/' + str(image_count)

        feature_dict = {path: tf.compat.v1.FixedLenFeature([], tf.string),
                        # 'file': tf.compat.v1.FixedLenFeature([], tf.string)
                        }

        features = tf.compat.v1.parse_single_example(serialized_example,
                                                     features=feature_dict)
        # label = tf.compat.v1.decode_raw(features['file'], tf.uint8)
        image_buffer = tf.reshape(features[path], shape=[])
        image = tf.compat.v1.decode_raw(image_buffer, tf.uint8)
        image = tf.reshape(image, tf.stack([height, width, num_depth]))
        image = tf.reshape(image, [1, height, width, num_depth])
        image_seq.append(image)

    image_seq = tf.concat(image_seq, 0)

    return image_seq  # , label


def load_videos(filenames, n_frames):
    """
    this function determines the number of videos available in all tfrecord files. It also checks on the correct shape of the single examples in the tfrecord
    files.
    :param filenames: a list, each entry containign a (relative) path to one tfrecord file
    :return: the number of overall videos provided in the filenames list
    """

    num_examples = 0

    if n_frames == 'all':
        n_frames_in_test_video = 354
    else:
        n_frames_in_test_video = n_frames

    videos = []
    # labels = []
    # create new session to determine batch_size for validation/test data
    with tf.compat.v1.Session() as sess_valid:
        filename_queue_val = tf.compat.v1.train.string_input_producer(
            filenames, num_epochs=1, shuffle=False)
        image_seq_tensor_val = read_and_decode(filename_queue_val, n_frames)

        init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                           tf.compat.v1.local_variables_initializer())
        sess_valid.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)
        try:
            while True:
                video = sess_valid.run(image_seq_tensor_val)
                # label = None # sess_valid.run(label_tensor_val)
                # assert np.shape(video) == (1, n_frames_in_test_video, height, width,
                #                            num_depth), "shape in the data differs from the expected shape"
                num_examples += 1
                videos.append(video)
                # labels.append(label)
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

    return videos  # , labels


def process(x, y):
    return x, y


def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=30)
    return embed.embed_file("animation.gif")


if __name__ == '__main__':
    # videos = load_videos(
    #     ['dataset/tfrecords/003-01-01.tfrecords', 'dataset/tfrecords/004-01-01.tfrecords', 'dataset/tfrecords/005-01-01.tfrecords'], 35)
    # print(np.shape(videos))
    # print(filename)

    # from glob import glob

    # from time import time

    # t = time()

    # # videos = load_videos(files, 5)

    # files = []
    # framesCounts = []
    # with open('dataset/split/rand-train.csv') as file:
    #     lines = file.readlines()
    #     for line in lines[1:]:
    #         line = line.splitlines()[0]
    #         file = line.split(',')[0]
    #         framesCount = int(line.split(',')[1])
    #         files.append(file)
    #         framesCounts.append(framesCount)

    # # print(framesCounts['011-08-21.mp4'])

    # # tfrecordsfiles = []
    # # for file, framesCount in tqdm(zip(files, framesCounts)):
    # #     tfrecordsfile = 'dataset/tfrecords/' + \
    # #         file.split('.')[0] + '.tfrecords'
    # #     tfrecordsfiles.append(tfrecordsfile)

    # tfrecordsfiles = glob('dataset/tfrecords/*')

    # # tfrecordsfiles.extend(tfrecordsfiles)
    # # tfrecordsfiles.extend(tfrecordsfiles)
    # # tfrecordsfiles.extend(tfrecordsfiles)
    # # tfrecordsfiles.extend(tfrecordsfiles)
    # # tfrecordsfiles.extend(tfrecordsfiles)

    # print(len(tfrecordsfiles))
    # videos = load_videos(tfrecordsfiles, 35)
    # videos = np.array(videos)

    # labels = [i for i in range(len(tfrecordsfiles))]
    # labels = np.array(labels)

    # print(len(videos))
    # print(np.shape(videos[0]))
    # print(type(videos))
    # print(type(videos[0]))
    # print(type(videos[0][0]))
    # print(type(labels))

    # ds = tf.data.Dataset.from_tensor_slices((videos, labels))
    # ds = ds.shuffle(buffer_size=10000)
    # ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.batch(batch_size=32)
    # ds = ds.prefetch(tf.data.AUTOTUNE)

    # print(time() - t)

    # print(np.shape(videos[0]))

    files = sorted(glob('asset/tfrecords/*'))
    print(files)
    videos = load_videos(files, 35)
    to_gif(np.array(videos[0]))
