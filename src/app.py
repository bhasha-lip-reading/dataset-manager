from glob import glob
import argparse
from Command.AudioExtractionCommand import AudioExtractionCommand
import config
from Command.AudioExtractionCommand import AudioExtractionCommand
from Command.CommandHandler import CommandHandler
from Command.VideoPlayerCommand import VideoPlayerCommand
from Command.AudioDetectionCommand import AudioDetectionCommand
from Command.AudioSeparationCommand import AudioSeparationCommand
from Command.VideoSeparationCommand import VideoSeparationCommand
from Command.AudioVideoMergeCommand import AudioVideoMergeCommand
from Command.FaceDetectionCommand import FaceDetectionCommand
from pathlib import Path
import Utilities.Utilities as util
# from DatasetProcessor import sampling
# from DatasetProcessor import TFRecordsGenerator, TFRecordsParser
parser = argparse.ArgumentParser()

parser.add_argument('--file')
parser.add_argument('--extract')
parser.add_argument('--plot')
parser.add_argument('--run')
parser.add_argument('--detect')
parser.add_argument('--separate')
parser.add_argument('--play')


def createDirectories(fileName):
    speaker, batch = util.getSpeakerBatch(fileName)

    Path("{}/S{}/B{}".format(config.AUDIO_SPLIT_DIR, speaker, batch)
         ).mkdir(parents=True, exist_ok=True)

    Path("{}/S{}/B{}".format(config.VIDEO_SPLIT_DIR, speaker, batch)
         ).mkdir(parents=True, exist_ok=True)

    Path("{}/S{}/B{}".format(config.MERGE_SPLIT_DIR, speaker, batch)
         ).mkdir(parents=True, exist_ok=True)

    Path("asset/frames").mkdir(parents=True, exist_ok=True)
    Path("asset/mouth-extracted-mute-video").mkdir(parents=True, exist_ok=True)
    Path("asset/mouth-extracted-video-with-sound").mkdir(parents=True, exist_ok=True)
    Path("asset/sampled-video").mkdir(parents=True, exist_ok=True)


def faceDetect(args):
    files = [
        "005-01.mp4",
        "005-02.mp4",
        "005-03.mp4",
        "005-04.mp4",
        "005-05.mp4",
        "005-06.mp4",
        "005-07.mp4",
        "005-08.mp4",
        "005-09.mp4",
        "005-10.mp4",
        "005-11.mp4",
        "005-12.mp4",
        "005-13.mp4",
        "005-14.mp4",
        "005-15.mp4",
        "005-16.mp4",
        "005-17.mp4",
        "005-18.mp4",
        "005-19.mp4",
        "005-20.mp4"
    ]

    from tqdm import tqdm

    for i in range(len(files)):
        print(files[i])
        file = files[i]
        args.file = file
 #    if(args.file == None):
 #        raise Exception('Missing: Requires filename.')

        createDirectories(args.file)
        faceDetector = FaceDetectionCommand(None)
        handler = CommandHandler(faceDetector)
        handler.handle(args)


if __name__ == "__main__":
    # time = [
    #     3.75, 4.90,
    #     6.20, 7.30,
    #     8.85, 9.90,
    #     11.60, 12.60,
    #     14.15, 15.20,
    #     16.70, 17.75,
    #     19.40, 20.45,
    #     21.95, 23.00,
    #     24.55, 25.65,
    #     27.00, 28.15,
    #     29.60, 30.80,
    #     32.20, 33.25,
    #     35.05, 36.05,
    #     37.45, 38.65,
    #     40.10, 41.15,
    #     42.70, 43.90,
    #     45.30, 46.35,
    #     47.85, 49.00,
    #     50.35, 51.45,
    #     52.95, 54.05,
    #     55.75, 56.70,
    #     58.20, 59.20,
    #     60.70, 62.20,
    #     63.35, 64.35,
    #     66.00, 67.05,
    # ]

    # i = 0
    # d = .55
    # while i + 1 < 50:
    # print('{:.02f},{:.02f}'.format(time[i] + d, time[i + 1] + d))
    # i += 2
    args = parser.parse_args()
    createDirectories(args.file)
    # faceDetect(args)

    """
    run for video splitting"""

    videoPlayer = VideoPlayerCommand(None)
    # faceDetector = FaceDetectionCommand(None)
    audioVideoMerger = AudioVideoMergeCommand(videoPlayer)
    videoSeparator = VideoSeparationCommand(audioVideoMerger)
    audioSeparator = AudioSeparationCommand(videoSeparator)
    audioDetector = AudioDetectionCommand(audioSeparator)
    audioExtractor = AudioExtractionCommand(audioDetector)

    handler = CommandHandler(audioExtractor)
#     handler = CommandHandler(faceDetector)
    handler.handle(args)

    #  sample videos to equal frame size"""
    # files = glob('./asset/mouth-extracted-video-with-sound/*.mp4')
    # sampling.sampleVideos(files)

    # TFRecordsGenerator.convert_videos_to_tfrecord(
    #     source_path='asset/sampled-video/',
    #     destination_path='asset/tfrecords/',
    #     n_videos_in_record=1,
    #     n_frames_per_video=35,
    #     dense_optical_flow=False,
    #     width=48,
    #     height=48,
    #     file_suffix="*.mp4")
