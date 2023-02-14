"""Module provide visluization class"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from librosa import display


class Visualizer:
    def __init__(self, data):
        print()
        print("Visualization object is created")
        self.data = data
        print()

    def visualize_random_samples(self, num_classes, num_samples):
        """
        This function plots the mfcc features
        ...
        Attributes
        ----------
        mfcc : A numpy array of mfcc features
        """
        label_samples = self.data["label"].sample(num_classes)
        random_samples = self.data.sample(num_samples)
        # plot for each label random samples
        for i in range(num_classes):
            fig = plt.figure(figsize=(10, 4.5 * num_samples))
            random_samples = self.data[self.data["label"] == label_samples.iloc[i]]
            for i in range(num_samples):
                ax = fig.add_subplot(num_samples, 1, i + 1)
                mfcc = random_samples.iloc[i]["mfcc_features"]
                label = random_samples.iloc[i]["label"]
                display.specshow(mfcc, x_axis="time", ax=ax)
                # plt.colorbar(ax=ax, format="%+2.f")
                ax.set_title(
                    "MFCC for sample number " + str(i) + " of class " + str(label)
                )
                plt.tight_layout()

        plt.show()
