from interface import implements
from AudioExtraction.Video import Video
import subprocess
import config


class Mp4Video(implements(Video)):
    def __init__(self, videoFilePath, audioFilePath):
        self.videoFilePath = videoFilePath
        self.audioFilePath = audioFilePath

    def extract(self):
        command = "ffmpeg -loglevel quiet -i {} -ab 160k -ac 2 -ar {} -vn {}".format(
            self.videoFilePath, config.DEFAULT_SAMPLING_RATE, self.audioFilePath)

        subprocess.call(command, shell=True)
