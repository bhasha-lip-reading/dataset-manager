import numpy as np
from scipy.io import wavfile


def read_audio_file(input_file):
    """
    This function returns a numpy array that stores the audio samples of a
    specified WAV of AIFF file
    """

    sampling_rate = 0
    signal = np.array([])
    if isinstance(input_file, str):
        sampling_rate, signal = wavfile.read(input_file)

    if signal.ndim == 2 and signal.shape[1] == 1:
        signal = signal.flatten()

    return sampling_rate, signal


def stereo_to_mono(signal):
    """
    This function converts the input signal
    (stored in a numpy array) to MONO (if it is STEREO)
    """

    if signal.ndim == 2:
        if signal.shape[1] == 1:
            signal = signal.flatten()
        else:
            if signal.shape[1] == 2:
                signal = (signal[:, 1] / 2) + (signal[:, 0] / 2)
    return signal
