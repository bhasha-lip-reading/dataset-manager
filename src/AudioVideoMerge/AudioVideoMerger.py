from glob import glob
import config
import subprocess
from Utilities import Utilities as util
from tqdm import tqdm
import moviepy.editor as editor
import os


class AudioVideoMerger:
    def __init__(self, fileName):
        self.fileName = fileName
        self.audioFilePath = util.getSeparatedAudioFilePath(self.fileName)
        self.videoFilePath = util.getSeparatedVideoFilePath(self.fileName)
        self.mergeVideoFilePath = util.getMergedVideoFilePath(self.fileName)

    def apply(self):
        for id in tqdm(range(config.WORDS_PER_BATCH), desc='Merge in progress:'):
            if os.path.exists(self.mergeVideoFilePath % (id + 1)):
                os.remove(self.mergeVideoFilePath % (id + 1))

            merge = 'ffmpeg -loglevel quiet -i {} -i {} -c:v copy -c:a aac {}'.format(
                self.videoFilePath % (id + 1), self.audioFilePath % (id + 1), self.mergeVideoFilePath % (id + 1))
            subprocess.call(merge, shell=True)

    def applyEditor(self):
        for id in tqdm(range(config.WORDS_PER_BATCH), desc='Merge in progress:'):
            video = editor.VideoFileClip(
                self.videoFilePath % (id + 1), audio_fps=False)

            audio = editor.AudioFileClip(self.audioFilePath % (
                id + 1), fps=config.DEFAULT_SAMPLING_RATE)

            audio = editor.CompositeAudioClip([audio])
            video = video.set_audio(audio)
            video.write_videofile(self.mergeVideoFilePath % (id + 1))
