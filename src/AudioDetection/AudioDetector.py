from config import INTERVAL_DIR, WORDS_PER_BATCH
from .Audio import Audio
from numpy import savetxt
import os
import scipy.io.wavfile as wavfile


class AudioDetector:
    def __init__(self, audio):
        self.audio: Audio = audio

    def saveAsCsv(self, intervals):
        csvFilePath = os.path.join(
            INTERVAL_DIR, self.audio.getAudioFileName().split('.')[0] + '.csv')

        if not os.path.isfile(csvFilePath):
            with open(csvFilePath, "w") as filePath:
                pass
        savetxt(csvFilePath, intervals, fmt='%.2f', delimiter=',')

    def apply(self, smoothingWindow=1.0, weight=0.3, plot=False):
        self.audio.read()
        intervals = self.audio.findIntervals(smoothingWindow, weight, plot)

        if len(intervals) != WORDS_PER_BATCH:
            print(
                'Error: In AudioDetector.py at line 26: Could not detect all the words.')

        self.saveAsCsv(intervals)
