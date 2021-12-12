from AudioExtraction.Video import Video


class AudioExtractor:
    def __init__(self, video: Video):
        self.video = video

    def apply(self):
        self.video.extract()
