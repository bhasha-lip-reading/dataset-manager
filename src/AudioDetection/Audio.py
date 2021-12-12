from interface import Interface


class Audio(Interface):
    def read(self):
        pass

    def getAudioFileName(self):
        pass

    def findIntervals(self, smoothingWindow, weight, plot):
        pass
