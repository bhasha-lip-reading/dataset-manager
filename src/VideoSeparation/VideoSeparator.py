from os import write
from Utilities import Utilities as util
import cv2
import numpy as np
from tqdm import tqdm
from FaceDetection import FaceDetector as detector


class VideoSeparator:
    def __init__(self, videoFileName):
        self.videoFileName = videoFileName
        self.videoFilePath = util.getVideoPath(videoFileName)

        self.capture = cv2.VideoCapture(self.videoFilePath)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fourcc = int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

    def getFrameId(self, t):
        return round(t * self.fps)

    def apply(self):

        intervalFileName = util.getIntervalPath(self.videoFileName)
        videoFilePath = util.getSeparatedVideoFilePath(self.videoFileName)
        intervals = np.genfromtxt(intervalFileName, delimiter=',')

        for i in tqdm(range(len(intervals)), desc='Video separation progress'):
            start = intervals[i][0]
            end = intervals[i][1]

            beginFrameId = self.getFrameId(start)
            endFrameId = self.getFrameId(end)

            writer = cv2.VideoWriter(
                videoFilePath % (i + 1), self.fourcc, self.fps, self.size)

            self.capture.set(cv2.CAP_PROP_POS_FRAMES, beginFrameId)

            ret = True
            frames = []
            while(self.capture.isOpened() and ret and writer.isOpened()):
                ret, frame = self.capture.read()
                frameId = self.capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
                if(frameId <= endFrameId):
                    writer.write(frame)
                    frames.append(frame)
                else:
                    break
            writer.release()
