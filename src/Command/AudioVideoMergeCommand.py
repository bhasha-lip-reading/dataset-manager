from Command.Command import Command
from AudioVideoMerge.AudioVideoMerger import AudioVideoMerger


class AudioVideoMergeCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        audioVideoMerger = AudioVideoMerger(args.file)
        audioVideoMerger.apply()
