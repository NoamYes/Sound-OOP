from information import Information
from processor import Processor

import pandas as pd


class SoundObjectOriented:
    """
    param train: train data will be used for modelling
    param test:  test data will be used for model evaluation
    """

    def __init__(self):
        # properties
        self.ntrain = None
        self.testID = None
        self.y_train = None
        self.train = None
        self.test = None
        self._info = Information()
        self._Preprocessor = Processor()

        print()
        print("SoundObjectOriented object is created")
        print()

    def add_data(self, train, test, index_name):
        # properties
        self.ntrain = train.shape[0]
        self.testID = test.reset_index()  # .drop("index", axis=1)["Id"]
        self.y_train = train["label"]  # .apply(lambda x: np.log1p(x))
        self.train = train.drop("label", axis=1)
        self.test = test

        # concatinating the whole data
        self.data = self.concat_data(self.train, self.test, index_name)
        self.orig_data = self.data.copy()
        print()
        print("Your data has been added")
        print()

    def concat_data(self, train, test, index_name):
        """
        concat train data and test data to make a data DataFrame
        :return:
        """
        data = pd.concat(
            [self.train.set_index(index_name), self.test.set_index(index_name)]
        ).reset_index(drop=True)

        return data

    # using the objects
    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        print(self._info.show_basic_info(self.data))

    def pre_processing(self):

        """
        preprocess the data before applying Ml algorithms
        """
        self.data = self._Preprocessor._process(self.data, self.ntrain)

        print()
        print("Data has been Pre-Processed")
        print()
