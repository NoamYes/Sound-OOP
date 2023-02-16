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
    N_MFCC = int(os.getenv("N_MFCC"))

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

    def reshape_mfcc(mfcc):
        return mfcc.reshape(-1, N_MFCC)

    train_extracted["mfcc_features"] = train_extracted["mfcc_features"].apply(
        reshape_mfcc
    )

    test_extracted["mfcc_features"] = test_extracted["mfcc_features"].apply(
        reshape_mfcc
    )

    # %% trim data for debug purposes

    # train_extracted = train_extracted[:40]
    # test_extracted = test_extracted[:40]

    # %% reduce dimensionality of data

    # train_extracted["mfcc_features"] = train_extracted["mfcc_features"].apply(
    #     lambda x: x[range(0, test_extracted.shape[1], 10)]
    # )
    # test_extracted["mfcc_features"] = test_extracted["mfcc_features"].apply(
    #     lambda x: x[range(0, test_extracted.shape[1], 10)]
    # )

    # %%

    sound_oop = SoundObjectOriented()
    sound_oop.add_data(train_extracted, test_extracted, index_name="fname")
    visualizer = sound_oop.visualizer(
        sound_oop,
    )
    # visualizer.visualize_random_samples(num_classes=4, num_samples=4)
    # sound_oop.information()
    sound_oop.pre_processing()
    # sound_oop.information()

    # load ml instace already created
    # ML = SoundObjectOriented.load_ml_instance("./assets/models/ML.pkl")

    ML = sound_oop.ml(sound_oop)
    ML.show_available_algorithms()
    # ML.init_classifiers(["Dummy Classifier Keras", "Cnn"])
    # ML.init_classifiers(["Dummy Classifier Keras"])
    ML.init_classifiers(["Dummy Classifier Sklearn"])
    # ML.init_classifiers(["Cnn"])
    # ML.init_classifiers("all")
    # ML.init_classifiers(["Dummy Classifier Keras", "Cnn"])

    # ML.train_test_validation()
    # ML.visualize_train_test()
    ML.cross_validation("all")
    # save the models
    ML.save_models("./assets/models/")
    # ML.load_models("./assets/models/")
    ML.visualize_cv(metrics=["accuracy"])
    ML.fit_best_model()
    ML.evaluate_best_model()
    ML.show_predictions()
    ML.save_predictions("predictions_best_model")

    SoundObjectOriented.persist_ml_instance(ML, "./assets/models/ML.pkl")


if __name__ == "__main__":
    main()

# %%
