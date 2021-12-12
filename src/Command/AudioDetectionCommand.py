from Command.Command import Command
from Utilities import Utilities as util
import os
from AudioDetection.AudioDetector import AudioDetector
from AudioDetection.WavAudio import WavAudio


def validate(args):
    if args.file == None:
        return False

    if args.run == None and args.detect != 'audio':
        return False

    return True


class AudioDetectionCommand(Command):
    def __init__(self, next=None):
        super().__init__(next)

    def handle(self, args):
        if validate(args):
            file = util.getAudioFilePath(args.file)
            target = util.getIntervalPath(args.file)

            if not os.path.exists(target):
                print('Detecting  audio from audio file: {}'.format(file))

                if args.plot == None:
                    plot = False

                audioDetector = AudioDetector(WavAudio(args.file))
                audioDetector.apply(plot=plot)
            else:
                print('Already exists interval file: {}'.format(target))
        else:
            print('Skipping audio detection.')
