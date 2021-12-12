from Command.Command import Command
from AudioExtraction.AudioExtractor import AudioExtractor
from AudioExtraction.Mp4Video import Mp4Video
from glob import glob
import os
from Utilities import Utilities as util


def validate(args):
    if args.file == None:
        return False

    if args.run == None and args.extract != 'audio':
        return False

    return True


class AudioExtractionCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        if validate(args):
            file = util.getVideoPath(args.file)
            target = util.getAudioFilePath(args.file)

            if not os.path.exists(target):
                print('Extracting  audio from video file: {}'.format(file))

                audioExtractor = AudioExtractor(
                    Mp4Video(file, target))
                audioExtractor.apply()
            else:
                print('Already exists audio file: {}'.format(target))
        else:
            print('Skipping audio extraction.')
