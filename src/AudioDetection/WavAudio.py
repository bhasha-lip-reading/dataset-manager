from AudioDetection.Audio import Audio
from interface import implements
import os
from config import AUDIO_DIR
from scipy.io import wavfile
from AudioDetection.audioSegmentation import silence_removal
from Utilities import Utilities as util


class WavAudio(implements(Audio)):
    def __init__(self, audioFileName):
        if len(audioFileName.split('/')) > 1:
            raise Exception(
                "Invalid audio file name. Valid example: 001-01.wav")

        self.audioFileName = audioFileName
        self.audioFilePath = util.getAudioFilePath(self.audioFileName)

        if not os.path.isfile(self.audioFilePath):
            raise FileNotFoundError(
                "Audio file '{}' not does not exists".format(self.audioFilePath))

    def read(self):
        self.samplingRate, self.audioData = wavfile.read(self.audioFilePath)
        return self.samplingRate, self.audioData

    def getAudioFileName(self):
        return self.audioFileName

    def findIntervals(self, smoothingWindow, weight, plot):
        intervals = silence_removal(
            self.audioData, self.samplingRate, 0.05, 0.05, smoothingWindow, weight, plot)
        return intervals
