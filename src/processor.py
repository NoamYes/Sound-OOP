"""Module provide Preprocessor class"""

import numpy as np
from pre_processing import PreProcessing


class Processor:
    def __init__(self):
        self.data = None
        self.y_train = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self._preprocessor = PreProcessing()

    def _process(self, data, ntrain):

        self.data = data
        self.ntrain = ntrain

        cols_drop = ["fname"]

        # Numeric columns
        num_cols = ["mfcc_features"]

        # Categorical columns
        cat_cols = ["label"]

        drop_strategies = [(cols_drop, 1)]

        # fill_strategies = # Standard Circuit Breakers & Romex

        # drop
        self.data = self._preprocessor.drop(self.data, drop_strategies)

        # fill nulls
        # self.data = self._preprocessor.fillna(self.ntrain, fill_strategies)

        # feature engineering
        # self.data = self._preprocessor.extract_features()

        # label encoder
        self.data = self._preprocessor.label_encoder(self.data, cat_cols)

        # extract y_train and y_test
        self.y_train = self.data["label"].to_numpy()[: self.ntrain]
        self.y_test = self.data["label"].to_numpy()[self.ntrain :]

        # get dummies
        self.data = self._preprocessor.get_dummies(cat_cols)

        # normalizing
        self.data = self._preprocessor.norm_data(num_cols)

        # extract X_train and X_test

        self.X_train = np.vstack(
            np.squeeze(self.data[: self.ntrain][num_cols].to_numpy())
        )
        self.X_test = np.vstack(
            np.squeeze(self.data[self.ntrain :][num_cols].to_numpy())
        )

        return self.data, self.y_train, self.X_train, self.X_test, self.y_test
