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
from utils.sound_features import get_mfcc_features_2


def main():
    """This is the main script of a dataScience project"""

    ENV = os.getenv("ENV")
    TRAIN_PATH = os.getenv("TRAIN_PATH")
    TEST_PATH = os.getenv("TRAIN_PATH")

    #%% load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test_post_competition.csv")

    #%% extract labels

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    #%% Load raw data and extract features
    prepare_data = PrepareData()
    train_extracted = prepare_data.extract_features(
        TRAIN_PATH, "train", loadPreComputed=False, save=True
    )
    test_extracted = prepare_data.extract_features(
        TEST_PATH, "test", loadPreComputed=False, save=True
    )
    # train_extracted = train["fname"].apply(get_mfcc_features_2, path=TRAIN_PATH)
    # print("done loading train mfcc")
    # test_extracted = test["fname"].apply(get_mfcc_features_2, path=TEST_PATH)
    # print("done loading test mfcc")
    y_train = train.loc[train_extracted.index.to_numpy()]
    #%%

    sound_oop = SoundObjectOriented()
    sound_oop.add_data(train_extracted, test_extracted, y_train, index_name="fname")
    # sound_oop.information()
    sound_oop.pre_processing()
    # sound_oop.information()

    ML = sound_oop.ml(sound_oop)
    ML.show_available_algorithms()
    ML.init_regressors("all")
    ML.train_test_validation()


if __name__ == "__main__":
    main()

# %%
