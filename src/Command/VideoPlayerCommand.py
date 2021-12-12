from Command.Command import Command
from VideoPlayer import VideoPlayer as player
from numpy import genfromtxt
from Utilities import Utilities as util


class VideoPlayerCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        if args.play == None:
            player.playAll(args.file)
