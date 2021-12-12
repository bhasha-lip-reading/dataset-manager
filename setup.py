from setuptools import setup
from pathlib import Path

root = Path(__file__).parent
with open(root/'README.md') as file:
    README = file.read()

REQUIREMENTS = [
    'autopep8==1.5.7',
    'certifi==2021.5.30',
    'charset-normalizer==2.0.3',
    'cycler==0.10.0',
    'decorator==4.4.2',
    'dlib==19.22.0',
    'ffmpeg==1.4',
    'ffmpeg-python==0.2.0',
    'ffpyplayer==4.3.2',
    'future==0.18.2',
    'idna==3.2',
    'imageio==2.9.0',
    'imageio-ffmpeg==0.4.4',
    'joblib==1.0.1',
    'kiwisolver==1.3.1',
    'matplotlib==3.4.2',
    'moviepy==1.0.3',
    'numpy==1.21.1',
    'opencv-python==4.5.3.56',
    'path==16.2.0',
    'Pillow==8.3.1',
    'proglog==0.1.9',
    'pycodestyle==2.7.0',
    'pyparsing==2.4.7',
    'python-dateutil==2.8.2',
    'python-interface==1.6.1',
    'requests==2.26.0',
    'scipy==1.7.0',
    'six==1.16.0',
    'threadpoolctl==2.2.0',
    'toml==0.10.2',
    'tqdm==4.61.2',
    'urllib3==1.26.6',
]

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Internet',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
]

setup(
    name='bhuiyans-dataset-builder',
    version='0.0.3',
    description='A video processing tool for visual data collection, end-to-end preprocessing, ready-to-go for model training.',
    long_description=README,
    url='https://github.com/MasumBhuiyan/dataset-builder',
    author='Masum Bhuiyan',
    author_email='masumbhuiyan577@gmail.com',
    license='Apache License',
    package_dir={'': 'src'},
    py_modules=['config', 'app'],
    include_package_data=True,
    packages=['AudioDetection',
              'AudioExtraction',
              'AudioPlayer',
              'AudioSeparation',
              'AudioVideoMerge',
              'Compression',
              'FaceDetection',
              'Preprocessing',
              'Utilities',
              'VideoPlayer',
              'VideoSeparation'],
    classifiers=CLASSIFIERS,
    install_requires=REQUIREMENTS,
    keywords='video preprocessing split merge video player'
)
