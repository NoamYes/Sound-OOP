"""Module provide PreProcessing class"""

import os
import pandas as pd
import numpy as np
from utils.sound_features import get_mfcc_features


class PrepareData:
    """
    This class prepares the data berfore applying ML
    """

    data: None

    def __init__(self):

        self.data = None
        print()
        print("pre-processing object is created")
        print()

    def extract_features(self, data_files_directory, data_name, loadPreComputed=False):
        """
        This function extracts the features you want from the raw data.
        ...
        Attributes
        ----------
        data : train directory
            The data path you want to load from
        features : A list of features to extract

        Returns
        ----------
        A new dataset with the extracted features.
        """
        if loadPreComputed and os.path.exists("data/extracted_features.csv"):
            extracted_features = pd.read_csv("data/extracted_features.csv", sep=",")
            extracted_features["mfcc_features"] = np.squeeze(
                extracted_features["mfcc_features"]
            ).str.split(",")
            return extracted_features
        self.data_directory = data_files_directory
        extracted_features = []
        file_names = []
        for filename in os.listdir(self.data_directory):
            f = os.path.join(self.data_directory, filename)
            # checking if it is a file
            if os.path.isfile(f):
                mfcc_features = get_mfcc_features(f)
                extracted_features.append(mfcc_features.reshape((-1, 1)))
                file_names.append(filename)
        # creating a dataframe from the extracted features and file names
        ex_dic = {"fname": file_names, "mfcc_features": extracted_features}
        cols = ["fname", "mfcc_features"]
        train_features_pd = pd.DataFrame(ex_dic, columns=cols)
        train_features_pd_cpy = train_features_pd.set_index("fname", inplace=False)
        train_features_pd_cpy.to_csv("data/" + data_name + "_extracted_features.csv")
        return train_features_pd

    # def drop(self, data, drop_strategies):
    #     """
    #     This function is used to drop a column or row from the dataset.
    #     ...
    #     Attributes
    #     ----------
    #     data : Pandas DataFrame
    #         The data you want to drop data from.
    #     drop_strategies : A list of tuples, each tuple has the data to drop,
    #     and the axis(0 or 1)

    #     Returns
    #     ----------
    #     A new dataset after dropping the unwanted data.
    #     """

    #     self.data = data

    #     for columns, ax in drop_strategies:
    #         if len(columns) == 1:
    #             self.data = self.data.drop(labels=column, axis=ax)
    #         else:
    #             for column in columns:
    #                 self.data = self.data.drop(labels=column, axis=ax)
    #     return self.data

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

    def label_encoder(self, columns):
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

        self.data[columns] = self.data[columns].apply(
            lambda x: lbl.fit_transform(x.astype(str)).astype(int)
        )

        return self.data

    # def get_dummies(self, columns):
    #     """
    #     This function is used to convert the data to dummies values.
    #     ...
    #     Attributes
    #     ----------
    #     data : Pandas DataFrame
    #         The data you want to convert.

    #     Returns
    #     ----------
    #     A dataset with dummies.
    #     """

    #     # convert our categorical columns to dummies
    #     for col in columns:
    #         dumm = pd.get_dummies(self.data[col], prefix=col, dtype=int)
    #         self.data = pd.concat([self.data, dumm], axis=1)

    #     self.data.drop(columns, axis=1, inplace=True)

    #     return self.data

    # def norm_data(self, columns):
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

    #     # Normalize our numeric data
    #     self.data[columns] = self.data[columns].apply(
    #         lambda x: np.log1p(x)
    #     )  # Normalize the data with Logarithms

    #     return self.data
