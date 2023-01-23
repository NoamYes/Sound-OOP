"""
Module docstring
"""
from dotenv import load_dotenv
import os
import pandas as pd
from information import Information
from pre_processing import PreProcessing


def main():
    """This is the main script of a dataScience project"""

    ENV = os.getenv("ENV")
    TRAIN_PATH = os.getenv("TRAIN_PATH")

    #%% load data
    train = pd.read_csv("data/train.csv")

    info = Information(train)
    info.show_basic_info(train)
    info.show_manual_info()

    pre_processing = PreProcessing(TRAIN_PATH)
    pre_processing.extractFeatures()
    #%%


if __name__ == "__main__":
    main()

# %%
