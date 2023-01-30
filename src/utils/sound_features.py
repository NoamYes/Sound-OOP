import librosa
import matplotlib.pyplot as plt
import os
import numpy as np


def get_mfcc_features(y, sr, n_mfcc=20, showWave=False, showFeatures=False):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if showWave:
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(y, sr=sr)
        plt.show()
    if showFeatures:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfcc, x_axis="time")
        plt.colorbar()
        plt.show()
    return mfcc


def truncate_or_pad_audio(audio, sample_rate):
    """
    This function is used to truncate or pad the audio to a specific length.
    ...
    Attributes
    ----------
    audio : The audio you want to truncate or pad.
    sample_rate : The sample rate of the audio.

    Returns
    ----------
    The truncated or padded audio.
    """
    target_length = int(os.getenv("AUDIO_LENGTH")) * sample_rate
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, max(0, target_length - len(audio))), "constant")
    return audio


def get_mfcc_features_2(
    self, name, path, n_mfcc=20, showWave=False, showFeatures=False
):
    y, sr = librosa.core.load(path + name)
    y = self.truncate_or_pad_audio(y, sr)
    mfcc_features = get_mfcc_features(y=y, sr=sr, n_mfcc=20)
    if showWave:
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(y, sr=sr)
        plt.show()
    if showFeatures:
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(mfcc_features, x_axis="time")
        plt.colorbar()
        plt.show()
    return mfcc_features
