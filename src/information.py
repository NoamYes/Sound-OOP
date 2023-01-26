"""Module provide information class"""

import numpy as np


class Information:
    """
    This class shows some information about the dataset
    """

    def __init__(self):

        print()
        print("Information object is created")
        print()

    def get_missing_values(self, data):
        """
        This function finds the missing values in the dataset
        ...
        Attributes
        ----------
        data : Pandas DataFrame
        The data you want to see information about

        Returns
        ----------
        A Pandas Series contains the missing values in descending order
        """
        # get the sum of all missing values in the dataset
        missing_values = data.isnull().sum()
        # sorting the missing values in a pandas Series
        missing_values = missing_values.sort_values(ascending=False)

        # returning the missing values Series
        return missing_values

    def show_basic_info(self, data):
        """
        This function shows some information about the data like
        Feature names,data type, number of missing values for each feature
        and ten samples of each feature
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to see information about

        Returns
        ----------
        Information about the DataFrame
        """
        self.data = data
        feature_dtypes = self.data.dtypes
        self.missing_values = self.get_missing_values(self.data)
        feature_names = self.missing_values.index.values
        rows, columns = data.shape

        print("=" * 50)
        print("====> This data contains {} rows and {} columns".format(rows, columns))
        print("=" * 50)
        print()

        print(
            "{:13} {:13} {:15}".format(
                "Feature Name".upper(),
                "Data Format".upper(),
                # "Null values(Num-Perc)".upper(),
                "Seven Samples".upper(),
            )
        )
        for feature_name, dtype in zip(feature_names, feature_dtypes[feature_names]):
            print(
                "{:15} {:14}".format(
                    feature_name,
                    str(dtype),
                    # str(missing_value)
                    # + " - "
                    # + str(round(100 * missing_value / sum(self.missing_values), 3))
                    # + " %",
                ),
                end="",
            )

            for i in np.random.randint(0, len(data), 7):
                print(data[feature_name].iloc[i], end=",")
            print()

        print("=" * 50)

        print(self.data.head(10))

    def show_manual_info(self):
        """
        This function shows the distribution and manually and not manually classified features
        ...
        Attributes
        ----------
        ----------
        Information about the DataFrame
        """
        category_group = self.data.groupby(["label", "manually_verified"]).count()
        plot = (
            category_group.unstack()
            .reindex(category_group.unstack().sum(axis=1).sort_values().index)
            .plot(
                kind="bar",
                stacked=True,
                title="Number of Audio Samples per Category",
                figsize=(16, 10),
            )
        )
        plot.set_xlabel("Category")
        plot.set_ylabel("Number of Samples")
