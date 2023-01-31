"""Module provide PreProcessing class"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import matplotlib as plt

from utils.sound_features import get_mfcc_features, truncate_or_pad_audio


class PrepareData:
    """
    This class prepares the data berfore applying ML
    """

    data: None

    def __init__(self):

        self.data = None
        print()
        print("pre-processing object is created")
        print()

    def extract_features(
        self,
        data_files_directory,
        data_name,
        loadPreComputed=False,
        save=False,
        save_path="data/",
    ):
        """
        This function extracts the features you want from the raw data.
        ...
        Attributes
        ----------
        data : train directory
            The data path you want to load from
        features : A list of features to extract

        Returns
        ----------
        A new dataset with the extracted features.
        """
        if loadPreComputed and os.path.exists("data/extracted_features.csv"):
            extracted_features = pd.read_csv("data/extracted_features.csv", sep=",")
            extracted_features["mfcc_features"] = np.squeeze(
                extracted_features["mfcc_features"]
            ).str.split(",")
            return extracted_features
        self.data_directory = data_files_directory
        extracted_features = []
        file_names = []
        for filename in tqdm(os.listdir(self.data_directory)):
            f = os.path.join(self.data_directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                y, sr = librosa.load(f, res_type="kaiser_fast")
                y = truncate_or_pad_audio(y, sr)
                mfcc_features = get_mfcc_features(y=y, sr=sr, n_mfcc=20)
                extracted_features.append(mfcc_features.reshape((-1, 1)).flatten())
                file_names.append(filename)
        # creating a dataframe from the extracted features and file names
        ex_dic = {"fname": file_names, "mfcc_features": extracted_features}
        cols = ["fname", "mfcc_features"]
        train_features_pd = pd.DataFrame(ex_dic, columns=cols)
        train_features_pd.set_index("fname", inplace=True)
        # creating a series from the extracted features and file names
        # series = pd.Series(extracted_features, index=file_names)
        if save:
            train_features_pd.to_csv(save_path + data_name + "_extracted_features.csv")
        return train_features_pd
