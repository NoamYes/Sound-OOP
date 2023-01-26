"""Module provide Preprocessor class"""

from pre_processing import PreProcessing


class Processor:
    def __init__(self, preProcessor):
        self.data = None
        self._preprocessor = preProcessor

    def _process(self, data, ntrain):

        self.data = data

        self.ntrain = ntrain

        cols_drop = ["Utilities", "OverallQual", "TotRmsAbvGrd"]

        # Numeric columns
        # num_cols = []

        # Categorical columns
        cat_cols = ["label"]

        # drop_strategies = [(cols_drop, 1)]

        # fill_strategies = # Standard Circuit Breakers & Romex

        # drop
        # self.data = self._preprocessor.drop(self.data, drop_strategies)

        # fill nulls
        # self.data = self._preprocessor.fillna(self.ntrain, fill_strategies)

        # feature engineering
        self.data = self._preprocessor.extract_features()

        # label encoder
        self.data = self._preprocessor.label_encoder(cat_cols)

        # normalizing
        #         self.data = self._preprocessor.norm_data(self.data, num_cols)

        # get dummies
        self.data = self._preprocessor.get_dummies(cat_cols)
        return self.data
