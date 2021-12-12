from AudioDetection.WavAudio import WavAudio
import scipy.io.wavfile as wavfile
from numpy import genfromtxt
from Utilities import Utilities as util
from tqdm import tqdm
import os
import config


class AudioSeparator:
    def __init__(self, audioFileName):
        self.audioFileName = audioFileName

        speaker, batch = util.getSpeakerBatch(audioFileName)
        path = '{}/S{}/B{}/'.format(config.AUDIO_SPLIT_DIR, speaker, batch)
        # path = '{}/'.format(config.AUDIO_SPLIT_DIR)

        if not os.path.exists(path):
            os.makedirs(path)

    def apply(self):
        samplingRate, audioData = WavAudio(self.audioFileName).read()

        intervalFileName = util.getIntervalPath(self.audioFileName)
        audioFilePath = util.getSeparatedAudioFilePath(self.audioFileName)

        """
        added for separation of audio customatically. Can be deleted later"""
        speaker, batch = util.getSpeakerBatch(self.audioFileName)
        label = (int(batch) - 1) * 25

        intervals = genfromtxt(intervalFileName, delimiter=',')

        for i in tqdm(range(len(intervals)), desc='Audio separation in progress'):
            # print(audioFilePath % (i + 1))
            wavfile.write(audioFilePath % (i + 1), samplingRate, audioData[int(
                samplingRate * intervals[i][0]): int(samplingRate * intervals[i][1])])
