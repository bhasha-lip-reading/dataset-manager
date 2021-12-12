import cv2
from ffpyplayer.player import MediaPlayer
from glob import glob
# from Utilities import Utilities as util
from tqdm import tqdm, trange
import time
import numpy as np
import os


MOUTH_EXTRACTED_VIDEO_MUTE_VIDEO_DIR = './asset/mouth-extracted-mute-video'
MOUTH_EXTRACTED_VIDEO_WITH_SOUND_DIR = './asset/mouth-extracted-video-with-sound'
WORDS_PER_BATCH = 25
AUGMENTATION_PER_SAMPLE = 6
SPEAKER_ID_LENGTH = 3
BATCH_ID_LENGTH = 2
WORDLIST_DIR = './asset/words'


def play(fileName):
    video = cv2.VideoCapture(fileName)
    player = MediaPlayer(fileName)

    while True:
        grabbed, frame = video.read()
        audioFrame, val = player.get_frame()
        if not grabbed:
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audioFrame is not None:
            image, t = audioFrame
    video.release()
    cv2.destroyAllWindows()


# def playAll(fileName):
#     mergedVideoFilePath = util.getMergedVideoFilePath(fileName)

#     words = []
#     with open(util.getWordsPath(fileName), encoding='utf8') as f:
#         for line in f:
#             word = line.strip()
#             words.append(word)

#     progressBar = trange(config.WORDS_PER_BATCH, desc='', leave=True)
#     for i in progressBar:
#         progressBar.set_description(
#             'Should say: {}'.format(words[i]), refresh=True)
#         play(mergedVideoFilePath % (i + 1))

def getSpeakerBatch(fileName):
    speaker = fileName[0:SPEAKER_ID_LENGTH]
    batch = fileName[SPEAKER_ID_LENGTH +
                     1: SPEAKER_ID_LENGTH + BATCH_ID_LENGTH + 1]
    return speaker, batch


def getWordsPath(file):
    speaker, batch = getSpeakerBatch(file)
    return os.path.join(WORDLIST_DIR, 'b' + batch + ".csv")


def playAll(fileName):
    speaker = int(fileName[:3])
    batch = int(fileName[4:6])

    words = []
    with open(getWordsPath(fileName), encoding='utf8') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    progressBar = trange(WORDS_PER_BATCH, desc='', leave=True)
    # for i in progressBar:
    #     dataPath = os.path.join(MOUTH_EXTRACTED_VIDEO_WITH_SOUND_DIR, '{:03d}-{:02d}-{:02d}-%02d.mp4'.format(
    #         speaker, batch, i + 1
    #     ))
    #     for sampleId in range(AUGMENTATION_PER_SAMPLE):
    #         progressBar.set_description(
    #             'Should say: {}'.format(words[i]), refresh=True)
    #         play(dataPath % (sampleId + 1))
    #         # pass
    for i in progressBar:
        dataPath = os.path.join('./asset/merged-video/S{:03d}/B{:02d}'.format(speaker, batch), '{:03d}-{:02d}-{:02d}.mp4'.format(
            speaker, batch, i + 1
        ))

        progressBar.set_description(
            'Should say: {}'.format(words[i]), refresh=True)
        play(dataPath)


if __name__ == '__main__':
    playAll("006-07.mp4")
