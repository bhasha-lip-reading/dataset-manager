import subprocess
from Utilities import Utilities as util


class Compressor:
    def __init__(self, fileName):
        self.fileName = fileName

    def applySizeCompression(self, factor):
        filePath = util.getUncompressedVideoPath(self.fileName)
        targetPath = util.getVideoPath(self.fileName)
        compressCommand = 'ffmpeg -loglevel quiet -i {} -vf \"scale=iw/{}:ih/{}\" {}'.format(
            filePath, factor, factor, targetPath)
        subprocess.call(compressCommand, shell=True)
