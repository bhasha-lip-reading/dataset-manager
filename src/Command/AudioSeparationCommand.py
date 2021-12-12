from Command.Command import Command
from Utilities import Utilities as util
import os
from AudioSeparation.AudioSeparator import AudioSeparator


def validate(args):
    if args.file == None:
        return False

    if args.run == None and args.separate != 'audio':
        return False

    return True


class AudioSeparationCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        if validate(args):
            audioSeparator = AudioSeparator(args.file)
            audioSeparator.apply()
        else:
            print('Skipping audio extraction.')
