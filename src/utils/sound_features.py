import librosa
import matplotlib.pyplot as plt


def get_mfcc_features(audio_file, n_mfcc=20, showWave=False, showFeatures=False):
    y, sr = librosa.load(audio_file)
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
