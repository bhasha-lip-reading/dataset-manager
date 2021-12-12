import dlib
import cv2
import os
from glob import glob
from tqdm import tqdm
import subprocess
from time import time
import numpy as np
import random

AUDIO_DIR = './asset/audio'
VIDEO_DIR = './asset/video'
INTERVAL_DIR = './asset/time'
UNCOMPRESSED_VIDEO_DIR = './asset/uncompressed-video'
PROCESSED_VIDEO_DIR = './asset/processed-video'
AUDIO_SPLIT_DIR = './asset/separated-audio'
VIDEO_SPLIT_DIR = './asset/separated-video'
WORDLIST_DIR = './asset/words'
MERGE_SPLIT_DIR = './asset/merged-video'
MOUTH_EXTRACTED_VIDEO_MUTE_VIDEO_DIR = './asset/mouth-extracted-mute-video'
MOUTH_EXTRACTED_VIDEO_WITH_SOUND_DIR = './asset/mouth-extracted-video-with-sound'

LANDMARK_DETECTOR_PATH = './src/FaceDetection/shape_predictor_68_face_landmarks.dat'
LIP_MARGIN = 0.3
LIP_CROP_SIZE = (300, 300)  # 480p

WORDS_PER_BATCH = 25
DEFAULT_SAMPLING_RATE = 44100  # Hz
SPEAKER_ID_LENGTH = 3
BATCH_ID_LENGTH = 2
(MEAN, STD) = (0.3891, 0.165)

FPS = 25
(FRAME_HEIGTH, FRAME_WIDTH) = (480, 360)
CROP_SIZE = (480, 360)

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(
    LANDMARK_DETECTOR_PATH
)


def shape2List(landmark):
    points = []
    for i in range(48, 68):  # 0, 68 for all, 48, 68 for lip
        points.append((landmark.part(i).x, landmark.part(i).y))
    return points


def detectLandmarkOpt(frame, lastLandmark):
    landmark = lastLandmark
    faces = faceDetector(frame, 1)

    if len(faces) < 1:
        print('Skipping {}: No face detected.')
    else:
        landmark = landmarkDetector(frame, faces[0])
        landmark = shape2List(landmark)
    return landmark


def extractLipOpt(frame, landmark):
    lip = landmark

    # Lip landmark sorted for determining lip region
    lip_x = sorted(lip, key=lambda pointx: pointx[0])
    lip_y = sorted(lip, key=lambda pointy: pointy[1])

    # Determine Margins for lip-only image
    x_add = int((-lip_x[0][0]+lip_x[-1][0]) * LIP_MARGIN)
    y_add = int((-lip_y[0][1]+lip_y[-1][1]) * LIP_MARGIN)

    crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add,
                lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image

    cropped = frame[crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]]
    cropped = cv2.resize(
        cropped, (LIP_CROP_SIZE[0], LIP_CROP_SIZE[1]), interpolation=cv2.INTER_CUBIC)

    return cropped


def frameProcessingOpt(filePath, targetPath):
    # lips = []
    capture = cv2.VideoCapture(filePath)
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

    writer = cv2.VideoWriter(targetPath, fourcc, fps, LIP_CROP_SIZE, 0)

    lastLandmark = None
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmark = detectLandmarkOpt(frame, lastLandmark)
        lip = extractLipOpt(frame, landmark)
        # lips.append(lip)
        lastLandmark = landmark

        writer.write(lip)
    writer.release()
    capture.release()
    # return lips


def readGrayFrames(filePath):
    frames = []
    capture = cv2.VideoCapture(filePath)

    while True:
        success, frame = capture.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    capture.release()
    return frames


def readAndResizeFrames(filePath):
    frames = []
    capture = cv2.VideoCapture(filePath)

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGTH),
                           interpolation=cv2.INTER_NEAREST)
        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)
    capture.release()
    return frames


def detectLandmark(file, frames):
    landmarks = []
    for i, frame in enumerate(frames):
        # faces = faceDetector(frame, 1)
        faces = faceDetector(frame)

        if len(faces) < 1:
            # print('Skipping {}: No face detected.'.format(file))
            if len(landmarks) > 0:
                landmarks.append(landmarks[-1])
            continue

        # if len(faces) > 1:
        #     print('Skipping {}: Too many face detected'.format(file))
        # if len(landmarks) > 0:
        #     landmarks.append(landmarks[-1])
        # continue

        for face in faces:
            landmark = landmarkDetector(frame, face)
            landmark = shape2List(landmark)
            landmarks.append(landmark)

    # assert len(landmarks) == 0, "no face detected for file {}".format(file)
    return landmarks


def extractLip(frames, landmarks):
    lips = []
    for i, landmark in enumerate(landmarks):
        # Landmark corresponding to lip
        lip = landmark

        # Lip landmark sorted for determining lip region
        lip_x = sorted(lip, key=lambda pointx: pointx[0])
        lip_y = sorted(lip, key=lambda pointy: pointy[1])

        # Determine Margins for lip-only image
        x_add = int((-lip_x[0][0]+lip_x[-1][0]) * LIP_MARGIN)
        y_add = int((-lip_y[0][1]+lip_y[-1][1]) * LIP_MARGIN)

        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add,
                    lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image

        cropped = frames[i][crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]]
        # print(cropped.shape)
        if(cropped.shape[0] == 0 or cropped.shape[1] == 0 or cropped.shape[2] == 0):
            continue
        cropped = cv2.resize(
            cropped, (LIP_CROP_SIZE[0], LIP_CROP_SIZE[1]), interpolation=cv2.INTER_CUBIC)        # Resize

        lips.append(cropped)
    return lips


def save(targetPath, lips):
    for i, lip in enumerate(lips):
        cv2.imwrite(targetPath + "%03d" % (i + 1) + ".jpg", lip)


"""
Data augmentations
"""


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class Rgb2Gray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def rgb2gray(self, rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return np.array(gray)

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([self.rgb2gray(_) for _ in frames], axis=0)
        return np.expand_dims(frames, axis=-1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        frames = np.array(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)
        # print("Center: ", frames.shape)
        t, h, w, c = frames.shape
        crop_h, crop_w = self.size
        delta_w = int(round((w - crop_w))/2.)
        delta_h = int(round((h - crop_h))/2.)
        frames = frames[:, delta_h:delta_h+crop_h, delta_w:delta_w+crop_w, :]
        return frames


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        frames = np.array(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)
        # print("Random crop: ", frames.shape)
        t, h, w, c = frames.shape
        # print("t, h, w, c", t, h, w, c)
        crop_h, crop_w = self.size
        delta_w = random.randint(0, 5)
        # delta_w = random.randint(0, w - crop_w)
        delta_h = random.randint(0, 5)
        # delta_h = random.randint(0, h - crop_h)
        frames = frames[:, delta_h:delta_h+crop_h, delta_w:delta_w+crop_w, :]
        # print(" = {}:{} {}:{}".format(
        # delta_h, delta_h + crop_h, delta_w, delta_w+crop_w))
        return frames


class HorizontalFlip(object):
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        frames = np.array(frames)
        # print(frames.shape)
        # print(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)

        # print(frames.shape)
        t, h, w, c = frames.shape
        # print("horizontal flip: ", frames.shape)

        if random.random() < self.flip_ratio:
            for index in range(t):
                ___ = frames[index]
                frames[index, :, :, 0] = cv2.flip(frames[index, :, :, 0], 1)
        # print("After flip: ", frames.shape)
        return frames
# def apply(speaker, batch):

#     filePath = '{}/S{}/B{}/*.mp4'.format(
#         MERGE_SPLIT_DIR, speaker, batch)

#     allFiles = sorted(glob(filePath))

#     for i in tqdm(range(len(allFiles)), desc='Lip extraction in progress'):
#         file = allFiles[i]
#         frames = readGrayFrames(file)
#         landmarks = detectLandmark(file, frames)
#         lips = extractLip(frames, landmarks)

#         fileName = file.split('/')[-1]
#         filePrefix = fileName.split('.')[0]

#         targetPath = '{}/S{}/B{}/{}/'.format(
#             PROCESSED_VIDEO_DIR, speaker, batch, filePrefix)

#         if not os.path.exists(targetPath):
#             os.makedirs(targetPath)

#         word = filePrefix.split('-')[-1]
#         for i, lip in enumerate(lips):
#             savedPath = targetPath + \
#                 "{}-{}-{}-%02d".format(speaker, batch,
#                                        word) % (i + 1) + ".jpg"
#             if not os.path.isfile(savedPath):
#                 cv2.imwrite(savedPath, lip)


def writeVideo(frames, filePath, targetPath):
    capture = cv2.VideoCapture(filePath
                               )
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))

    writer = cv2.VideoWriter(targetPath, fourcc, fps,
                             LIP_CROP_SIZE, 0)

    for frame in frames:
        # cv2.imshow(filePath, frame)
        # cv2.waitKey(2)
        writer.write(frame)
    writer.release()


def preprocess(filename):
    # t = time()
    processedPath = './asset/processed-all/{}.mp4'.format(filename)
    videoPath = './asset/video-all/{}.mp4'.format(filename)
    audioPath = './asset/audio-all/{}.wav'.format(filename)
    dataPath = './asset/dataset/{}.mp4'.format(filename)

    frames = readGrayFrames(videoPath)
    landmarks = detectLandmark(videoPath, frames)
    lips = extractLip(frames, landmarks)
    # print(landmarks[0])

    # frameProcessingOpt(videoPath, processedPath)

    writeVideo(frames, videoPath, processedPath)

    merge = 'ffmpeg -loglevel quiet -i {} -i {} -c:v copy -c:a aac {}'.format(
        processedPath, audioPath, dataPath)
    subprocess.call(merge, shell=True)

    # print('execution time: {} secs'.format(time() - t))


def augment(frames):
    videos = []

    # horizontal = HorizontalFlip(1.0).__call__(frames)
    # centerCropHorizontal = CenterCrop(CROP_SIZE).__call__(horizontal)
    # randomCropHorizontal = RandomCrop(CROP_SIZE).__call__(horizontal)

    # centerCropOriginal = CenterCrop(CROP_SIZE).__call__(frames)
    # randomCropOriginal = RandomCrop(CROP_SIZE).__call__(frames)

    # iter = 5
    # while iter > 0:
    videos.append(frames)
    # videos.append(centerCropOriginal)
    # videos.append(randomCropOriginal)
    # videos.append(horizontal)
    # videos.append(randomCropHorizontal)
    # videos.append(centerCropHorizontal)
    # videos.append(randomCropOriginal)
    # iter -= 1

    # videos.extend([frames, frames, frames, frames, frames])
    # videos.extend([centerCropOriginal, centerCropOriginal])
    # videos.extend(randomCropOriginal)
    # videos.extend(horizontal)

    # videos.extend(centerCropHorizontal)
    # videos.extend(randomCropOriginal)
    # iter -= 1
    return videos


"""
saving file in mouth-extracted-mute-video directory
format: speaker-batch-wordId-sampleId.mp4
"""


def apply(speaker, batch):
    speaker = int(speaker)
    batch = int(batch)

    fileFormat = '{:03d}-{:02d}-%02d-%02d'.format(int(speaker), int(batch))
    # print(fileFormat % (1, 0))

    for wordId in tqdm(range(WORDS_PER_BATCH), desc='Augmentation & mouth extraction in progress'):
        videoPath = 'S{:03d}/B{:02d}/{:03d}-{:02d}-{:02d}.mp4'.format(
            speaker, batch, speaker, batch, wordId + 1)
        videoPath = os.path.join(MERGE_SPLIT_DIR, videoPath)

        audioPath = 'S{:03d}/B{:02d}/{:03d}-{:02d}-{:02d}.wav'.format(
            speaker, batch, speaker, batch, wordId + 1)
        audioPath = os.path.join(AUDIO_SPLIT_DIR, audioPath)

        processedPath = os.path.join(MOUTH_EXTRACTED_VIDEO_MUTE_VIDEO_DIR, '{:03d}-{:02d}-{:02d}-%02d.mp4'.format(
            speaker, batch, wordId + 1
        ))

        dataPath = os.path.join(MOUTH_EXTRACTED_VIDEO_WITH_SOUND_DIR, '{:03d}-{:02d}-{:02d}-%02d.mp4'.format(
            speaker, batch, wordId + 1
        ))

        frames = readAndResizeFrames(videoPath)
        # print("shape", np.array(frames).shape)
        # print("frame shape", frames[0].shape)
        # print(dataPath)

        augmentedVideos = augment(frames)

        sampleId = 0
        for frames in augmentedVideos:
            landmarks = detectLandmark(videoPath, frames)
            lips = extractLip(frames, landmarks)

            if len(lips) == 0:
                continue

            # print(np.asarray(lips).shape)

            iter = 5
            while iter > 0:
                writeVideo(lips, videoPath, processedPath % (sampleId + 1))
                iter -= 1

                merge = 'ffmpeg -loglevel quiet -i {} -i {} -c:v copy -c:a aac {}'.format(
                    processedPath % (sampleId + 1), audioPath, dataPath % (sampleId + 1))
                subprocess.call(merge, shell=True)

                sampleId += 1

        # assert sampleId != 0, "mouth extraction failed on {}".format(videoPath)


if __name__ == '__main__':

    apply(31, 6)
    print("Done")
    # audioFiles=sorted(glob('./asset/audio-all/*'))
    # videoFiles=sorted(glob('./asset/video-all/*'))

    # preprocess('007-06-03')

    # speakers=['003', '004', '005', '006', '007',
    #             '011', '013', '014', '015', '017', '020']

    # batches=['01', '02', '03', '04', '05', '06', '07', '08']

    # for batch in tqdm(batches):
    #     for i in range(25):
    #         filename='020-{}-{:02d}'.format(batch, i + 1)
    #         preprocess(filename)

    # for speaker in speakers:
    #     files = glob(
    #         '/Users/bhuiyans/Desktop/private-apps/dataset-builder/asset/separated-video/S{}/*/*'.format(speaker))

    #     for file in files:
    #         subprocess.call(
    #             'cp -i {} /Users/bhuiyans/Desktop/private-apps/dataset-builder/asset/video-all/'.format(file), shell=True)

    #     print(len(files))

    # has problems: 007-06-03,
