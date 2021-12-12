import cv2
from time import time
from tqdm import tqdm
from glob import glob
import numpy as np
import os

MAX_SEQ_LENGTH = 35
IMG_SIZE = 300


def downsample(framesOriginal):
    totalFrames = len(framesOriginal)
    framesToBeDeleted = totalFrames - MAX_SEQ_LENGTH
    takes = totalFrames // framesToBeDeleted

    if totalFrames == MAX_SEQ_LENGTH:
        return framesOriginal

    frames = []
    lastFrame = None
    for i in range(totalFrames):
        if (i + 1) % takes == 0:
            continue
        frames.append(framesOriginal[i])
        lastFrame = framesOriginal[i]
        if len(frames) == MAX_SEQ_LENGTH:
            break

    # assert len(frames) < MAX_SEQ_LENGTH, 'Length: {}'.format(len(frames))

    while len(frames) < MAX_SEQ_LENGTH:
        frames.append(lastFrame)

    assert len(frames) == MAX_SEQ_LENGTH, 'Does not satisfy frame count'
    return frames


def upsample(framesOriginal):
    totalFrames = len(framesOriginal)
    upsample_rate = MAX_SEQ_LENGTH // totalFrames + 1

    frames = []
    for i in range(totalFrames):
        for j in range(upsample_rate):
            frames.append(framesOriginal[i])
    assert len(frames) >= MAX_SEQ_LENGTH, 'Frames upsampling fails'
    return downsample(frames)


def reduceFrames(framesOriginal):
    frames = []
    for i in range(len(framesOriginal)):
        if i % 2 == 0:
            frames.append(framesOriginal[i])
    return frames


def getVideo(file, targetPath):
    capture = cv2.VideoCapture(file)

    framesOriginal = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        framesOriginal.append(frame)

    frames = []
    totalFrames = len(framesOriginal)

    if totalFrames > MAX_SEQ_LENGTH:
        frames = downsample(framesOriginal)
    elif totalFrames < MAX_SEQ_LENGTH:
        frames = upsample(framesOriginal)
    else:
        frames = framesOriginal

    assert len(frames) == MAX_SEQ_LENGTH
    writeVideo(frames, file, targetPath)
    capture.release()


def writeVideo(frames, filePath, targetPath):
    capture = cv2.VideoCapture(filePath)
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

    writer = cv2.VideoWriter(
        targetPath, fourcc, int(fps), (IMG_SIZE, IMG_SIZE), 0)

    for frame in frames:
        writer.write(frame)
    writer.release()
    capture.release()


"""
Input files format: ./asset/mouth-extracted-video-with-sound/003-01-01-01.mp4
"""


def sampleVideos(files):
    for file in tqdm(files):
        targetPath = os.path.join('asset/sampled-video/', file.split('/')[-1])
        getVideo(file, targetPath)


if __name__ == '__main__':
    # t = time()
    # speaker = '001'
    # """
    #  sample videos to equal frame size"""
    # files = sorted(glob('./asset/mouth-extracted/{}/*.mp4'.format(speaker)))

    # print("Sample size: ", len(files))
    # sampleVideos(files)

    # print(time() - t)


    files = []
    for speaker in range(16):
        for batch in range(4):
            f = sorted(glob('./asset/mouth-extracted/{:03d}/{:03d}-{:02d}-*.mp4'.format(speaker + 1, speaker + 1, 16 + batch + 1)))
            files.extend(f)
    sampleVideos(files)