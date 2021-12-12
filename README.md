# Dataset-builder ![Build](https://img.shields.io/badge/Design-passing-Green)
A video processing tool for visual data collection, end-to-end preprocessing, ready-to-go for model training.

### Input
A video or a set of videos. Currently supports ```.mp4``` only. 

### Quick install

```

pip install bhuiyans-dataset-builder

```

### Features

* Audio extraction :v:
  - ```python src/app.py --fileName 001-01.mp4 --extractAudio yes```
* Audio detection :v:
  - ```python src/app.py --fileName 001-01.wav --detectAudio yes```
  - To show plot: ```python src/app.py --fileName 001-01.wav --detectAudio yes --plot true```
* Audio-video split :v:
  - To separate audio: ```python src/app.py --fileName 001-01.wav --separateAudio yes```
  - To separate video: ```python src/app.py --fileName 001-01.mp4 --separateVideo yes```
* Audio-video merge :v:
  - ```python src/app.py --fileName 001-01.mp4 --mergeAudioVideo yes```
* Video player :v:
  - Play the separated videos to see all is good: ```python src/app.py --fileName 001-01.mp4 --playAll```
* Video compression :v:
  - ```python src/app.py --fileName 001-01.mp4 --compressBySize 2```
* Face detection and region extraction :v:
  - ```python src/app.py --fileName 001-01.mp4 --detectLip yes --speaker 001```
* Video data processing utilities :v:

### How to use it?

* Clone the repository: ```https://github.com/MasumBhuiyan/visual-data-manager.git```
* Open terminal and <b>cd</b> to the directory where <b>requirements.txt</b> is located.
* Create, activate, and install packages in a virtual environment
  - ```pip install virtualenv```
  - ```virtualenv env```
  - ```source env/bin/activate```
  - ```pip install -r requirements.txt```
* To split a video run: ```will be updated```
