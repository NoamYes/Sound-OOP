"""Module provide Preprocessor class"""

from pre_processing import PreProcessing


class Processor:
    def __init__(self):
        self.data = None
        self.y = None
        self._preprocessor = PreProcessing()

    def _process(self, data, y, ntrain):

        self.data = data
        self.y = y
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
        self.y = self._preprocessor.label_encoder(self.y, cat_cols)

        # normalizing
        # self.data = self._preprocessor.norm_data(num_cols)

        # get dummies
        # self.data = self._preprocessor.get_dummies(cat_cols)
        return self.data, self.y
