from Command.Command import Command
from VideoSeparation.VideoSeparator import VideoSeparator


class VideoSeparationCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        videoSeparator = VideoSeparator(args.file)
        videoSeparator.apply()
