from Command.Command import Command
from FaceDetection import FaceDetector as detector
from Utilities import Utilities as util


class FaceDetectionCommand(Command):
    def __init__(self, next):
        super().__init__(next)

    def handle(self, args):
        speaker, batch = util.getSpeakerBatch(args.file)
        detector.apply(speaker, batch)
