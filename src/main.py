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
    DATA_PATH = os.getenv("DATA_PATH")

    # %% load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test_post_competition.csv")

    # %% extract labels

    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    # %% Load raw data and extract features
    prepare_data = PrepareData()
    train_extracted = prepare_data.extract_features(
        TRAIN_PATH, "train", loadPreComputed=True, save=True, save_path=DATA_PATH + "/"
    )
    test_extracted = prepare_data.extract_features(
        TEST_PATH, "test", loadPreComputed=True, save=True, save_path=DATA_PATH + "/"
    )
    train_extracted["label"] = train.loc[train_extracted["fname"].to_numpy()][
        "label"
    ].to_numpy()
    test_extracted["label"] = test.loc[test_extracted["fname"].to_numpy()][
        "label"
    ].to_numpy()

    train_extracted.set_index("fname", inplace=True)
    test_extracted.set_index("fname", inplace=True)

    # %% trim data for debug purposes

    train_extracted = train_extracted[:10]
    test_extracted = test_extracted[:10]

    # %%

    sound_oop = SoundObjectOriented()
    sound_oop.add_data(train_extracted, test_extracted, index_name="fname")
    # sound_oop.information()
    sound_oop.pre_processing()
    # sound_oop.information()

    ML = sound_oop.ml(sound_oop)
    ML.show_available_algorithms()
    ML.init_regressors("all")
    ML.train_test_validation()
    ML.visualize_train_test()
    ML.cross_validation("all")
    ML.visualize_cv()


if __name__ == "__main__":
    main()

# %%
