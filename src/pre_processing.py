"""Module provide PreProcessing class"""

import os
from utils.sound_features import get_mfcc_features

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class PreProcessing:
    """
    This class prepares the data berfore applying ML
    """

    data: None

    def __init__(self):

        self.data = None
        print()
        print("pre-processing object is created")
        print()

    def drop(self, data, drop_strategies):
        """
        This function is used to drop a column or row from the dataset.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to drop data from.
        drop_strategies : A list of tuples, each tuple has the data to drop,
        and the axis(0 or 1)

        Returns
        ----------
        A new dataset after dropping the unwanted data.
        """

        self.data = data

        # for columns, ax in drop_strategies:
        #     if len(columns) == 1:
        #         self.data = self.data.drop(labels=columns[0], axis=ax)
        #     else:
        #         for column in columns:
        #             self.data = self.data.drop(labels=column, axis=ax)
        return self.data

    # def fillna(self, ntrain, fill_strategies):
    #     """
    #     This function fills NA/NaN values in a specific column using a specified method(zero,mean,...)
    #     ...
    #     Attributes
    #     ----------
    #     data : Pandas DataFrame
    #         The data you want to impute its missing values
    #     fill_strategies : A dictionary, its keys represent the columns,
    #     and the values represent the value to use to fill the Nulls.

    #     Returns
    #     ----------
    #     A new dataset without null values.
    #     """

    #     def fill(column, fill_with):

    #         if str(fill_with).lower() in ["zero", 0]:
    #             self.data[column].fillna(0, inplace=True)
    #         elif str(fill_with).lower() == "mode":
    #             self.data[column].fillna(self.data[column].mode()[0], inplace=True)
    #         elif str(fill_with).lower() == "mean":
    #             self.data[column].fillna(self.data[column].mean(), inplace=True)
    #         elif str(fill_with).lower() == "median":
    #             self.data[column].fillna(self.data[column].median(), inplace=True)
    #         else:
    #             self.data[column].fillna(fill_with, inplace=True)

    #         return self.data

    #     # LotFrontage: Linear feet of street connected to property
    #     self.data["LotFrontage"] = (
    #         self.data.groupby("Neighborhood")["LotFrontage"]
    #         .apply(lambda x: x.fillna(x.median()))
    #         .values
    #     )

    #     # Meaning that NO Masonry veneer
    #     self.data["MSZoning"] = self.data["MSZoning"].transform(
    #         lambda x: x.fillna(x.mode().values[0])
    #     )

    #     # imputing columns according to its strategy
    #     for columns, strategy in fill_strategies:
    #         if len(columns) == 1:
    #             fill(columns[0], strategy)
    #         else:
    #             for column in columns:
    #                 fill(column, strategy)

    #     return self.data

    # def feature_engineering(self):
    #     """
    #     This function is used to apply some feature engineering on the data.
    #     ...
    #     Attributes
    #     ----------
    #     data : Pandas DataFrame
    #         The data you want to apply feature engineering on.

    #     Returns
    #     ----------
    #     A new dataset with new columns and some additions.
    #     """
    #     # creating new columns
    #     self.data["TotalSF"] = (
    #         self.data["TotalBsmtSF"] + self.data["1stFlrSF"] + self.data["2ndFlrSF"]
    #     )

    #     # Convert some columns from numeric to string
    #     self.data[["YrSold", "MSSubClass", "MoSold", "OverallCond"]] = self.data[
    #         ["YrSold", "MSSubClass", "MoSold", "OverallCond"]
    #     ].astype(str)

    #     # Convert some columns from numeric to int
    #     self.data[
    #         [
    #             "BsmtHalfBath",
    #             "BsmtFinSF1",
    #             "BsmtFinSF2",
    #             "BsmtFullBath",
    #             "BsmtUnfSF",
    #             "GarageCars",
    #             "GarageArea",
    #         ]
    #     ] = self.data[
    #         [
    #             "BsmtHalfBath",
    #             "BsmtFinSF1",
    #             "BsmtFinSF2",
    #             "BsmtFullBath",
    #             "BsmtUnfSF",
    #             "GarageCars",
    #             "GarageArea",
    #         ]
    #     ].astype(
    #         int
    #     )

    #     return self.data

    def label_encoder(self, df, columns):
        """
        This function is used to encode the data to categorical values to benefit from increasing or
        decreasing to build the model
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to encode.
        columns : columns to convert.

        Returns
        ----------
        A dataset without categorical data.
        """

        # Convert all categorical collumns to numeric values
        lbl = LabelEncoder()

        df[columns] = df[columns].apply(
            lambda x: lbl.fit_transform(x.astype(str)).astype(int)
        )

        return df

    def get_dummies(self, columns):
        """
        This function is used to convert the data to dummies values.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to convert.

        Returns
        ----------
        A dataset with dummies.
        """

        # convert our categorical columns to dummies
        for col in columns:
            dumm = pd.get_dummies(self.data[col], prefix=col, dtype=int)
            self.data = pd.concat([self.data, dumm], axis=1)

        self.data.drop(columns, axis=1, inplace=True)

        return self.data

    def norm_data(self, col):
        #     """
        #     This function is used to normalize the data.
        #     ...
        #     Attributes
        #     ----------
        #     data : Pandas DataFrame
        #         The data you want to normalize.

        #     Returns
        #     ----------
        #     A new normalized dataset.
        #     """

        # max = np.vstack(np.squeeze(self.data[columns].to_numpy())).max()
        # min = np.vstack(np.squeeze(self.data[columns].to_numpy())).min()
        # #     # Normalize our numeric data
        # self.data[columns] = self.data[columns].apply(
        #     lambda arr: (arr - min) / (max - min + 1e-6) - 0.5
        # )
        # mean = np.vstack(np.squeeze(self.data[columns].to_numpy())).mean()
        # std = np.vstack(np.squeeze(self.data[columns].to_numpy())).std()

        mean = np.stack(self.data["mfcc_features"].to_list()).mean(axis=0)
        std = np.stack(self.data["mfcc_features"].to_list()).std(axis=0)

        def normalize_arr(arr):
            return np.divide((arr - mean), (std + 1e-6))

        #     # Normalize our numeric data
        self.data[col] = self.data[col].apply(
            normalize_arr
        )  # Normalize the data with Logarithms

        # def minmax_normalize(elem, min, max):
        #     """
        #     Scale data in range [0, 1]
        #     Input: dile column features
        #     """
        #     elem = (elem - min) / (max - min + 1e-6)
        #     return elem - 0.5

        # normalized_data = self.data.apply(minmax_normalize, axis=1)
        return self.data

    def flatten_data(self, col):
        """
        This function is used to flatten the data.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to flatten.

        Returns
        ----------
        A new flattened dataset.
        """

        self.data[col] = self.data[col].apply(lambda arr: arr.flatten())

        return self.data

    def under_sample_features(self, col, under_rate=10):
        """
        This function is used to under sample the data.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to under sample.

        Returns
        ----------
        A new under sampled dataset.
        """

        self.data[col] = self.data[col].apply(
            lambda arr: arr[range(1, len(arr), under_rate)]
        )

        return self.data

    def average_frame_features(self, col):
        """
        This function is used to average the data.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to average.

        Returns
        ----------
        A new averaged dataset.
        """

        self.data[col] = self.data[col].apply(lambda arr: np.mean(arr, axis=0))

        return self.data
