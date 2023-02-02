from information import Information
from ml import ML
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
        self.X_train = None
        self.train = None
        self.test = None
        self._info = Information()
        self._processor = Processor()

        print()
        print("SoundObjectOriented object is created")
        print()

    def add_data(self, X_train, X_test, index_name):
        # properties
        self.ntrain = X_train.shape[0]
        self.testID = X_test.reset_index()  # .drop("index", axis=1)["Id"]
        self.train = X_train  # .drop("label", axis=1)
        self.test = X_test

        # concatinating the whole data
        # self.data = self.concat_data(self.train, self.test, index_name)
        # self.orig_data = self.data.copy()
        self.data = X_train
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
        self.data, self.y_train, self.X_train = self._processor._process(
            self.data, self.ntrain
        )
        print()
        print("Data has been Pre-Processed")
        print()
        return self.data, self.y_train, self.X_train

    class visualizer:
        def __init__(self, House_Price_OOP):

            self.hp = House_Price_OOP
            self.data = self.hp.data
            self.y_train = self.hp.y_train
            self.ntrain = self.hp.ntrain
            self.testID = self.hp.testID
            self.data_vis = data_visualization

        def box_plot(self, columns):

            self.data_vis.box_plot(columns)

        def bar_plot(self, columns):

            self.data_vis.bar_plot(columns)

    class ml:
        def __init__(self, House_Price_OOP):

            self.hp = House_Price_OOP
            self.data = self.hp.data
            self.X_train = self.hp.X_train
            self.y_train = self.hp.y_train
            self.ntrain = self.hp.ntrain
            self.testID = self.hp.testID
            self._ML_ = ML(
                data=self.X_train,
                y_train=self.y_train,
                testID=self.testID,
                test_size=0.2,
                ntrain=self.ntrain,
            )

        def show_available_algorithms(self):

            self._ML_.show_available()

        def init_regressors(self, num_models="all"):

            self._ML_.init_ml_regressors(num_models)

        def train_test_validation(self, show_results=True):

            self._ML_.train_test_eval_show_results(show=show_results)

        def cross_validation(self, num_models=4, n_folds=5, show_results=False):

            self._ML_.cv_eval_show_results(
                num_models=num_models, n_folds=n_folds, show=show_results
            )

        def visualize_train_test(
            self, metrics=["r_squared", "adjusted r_squared", "mae", "mse", "rmse"]
        ):

            self._ML_.visualize_results(cv_train_test="train test", metrics=metrics)

        def visualize_cv(self, metrics=["r_squared", "rmse"]):

            self._ML_.visualize_results(cv_train_test="cv", metrics_cv=metrics)

        def fit_best_model(self):

            self._ML_.fit_best_model()

        def show_predictions(self):

            return self._ML_.show_predictions()

        def save_predictions(self, file_name):

            self._ML_.save_predictions(file_name)
            print("The prediction is saved")
