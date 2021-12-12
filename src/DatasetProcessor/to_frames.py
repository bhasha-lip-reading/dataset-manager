import cv2
import os
from glob import glob
from tqdm import tqdm

IMG_SIZE = 48


def read_video(file, save_to):
    capture = cv2.VideoCapture(file)

    if not os.path.exists(save_to):
        os.makedirs(save_to)  # creating TFRecords output folder

    i = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(save_to + '/frame-%02d.jpg' % i, frame)
        i += 1


if __name__ == '__main__':

    files = sorted(glob('asset/sampled-video/*'))
    # print(files)

    for file in tqdm(files, desc='conversion into frames'):
        save_to = 'asset/frames/' + file.split('/')[-1][:-4]
        read_video(file, save_to)

