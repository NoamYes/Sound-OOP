"""
Module docstring
"""
from dotenv import load_dotenv
import os
import pandas as pd
from information import Information
from pre_processing import PreProcessing
from prepare_data import PrepareData
from sound_oop import SoundObjectOriented


def main():
    """This is the main script of a dataScience project"""

    ENV = os.getenv("ENV")
    TRAIN_PATH = os.getenv("TRAIN_PATH")
    TEST_PATH = os.getenv("TRAIN_PATH")

    #%% load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test_post_competition.csv")

    #%% Load raw data and extract features
    prepare_data = PrepareData()
    train_extracted = prepare_data.extract_features(TRAIN_PATH, loadPreComputed=False)

    #%%

    sound_oop = SoundObjectOriented()
    sound_oop.add_data(train, test, index_name="fname")
    sound_oop.information()
    sound_oop.pre_processing()
    sound_oop.information()


if __name__ == "__main__":
    main()

# %%
