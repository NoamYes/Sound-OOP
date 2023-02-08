import numpy as np
import pandas as pd

# visualization libraries
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode

pd.options.plotting.backend = "plotly"

# machine learning libraries:
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_validate,
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
import xgboost as xgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier

# from scikeras.wrappers import KerasClassifier


class ML:
    def __init__(
        self, data, X_train, y_train, X_test, testID, test_size, ntrain, nClasses=4
    ):
        print()
        print("Machine Learning object is created")
        print()

        self.data = data
        self.ntrain = ntrain
        self.test_size = test_size
        self.X_train = X_train
        self.test = X_test
        self.testID = testID
        self.y_train = y_train[: self.ntrain]
        self.input_shape = (self.X_train.shape[1],)

        self.reg_models = {}
        self.base_models = {}

        def build_dummy_model():
            Input(self.input_shape)
            model = Sequential()
            model.add(Dense(nClasses))
            # Compile the model
            model.compile(
                loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
            )

            return model

        def build_model_graph():
            Input(self.input_shape)
            model = Sequential()
            model.add(Dense(256))
            model.add(Activation("relu"))
            model.add(Dropout(0.5))
            model.add(Dense(256))
            model.add(Activation("relu"))
            model.add(Dropout(0.5))
            model.add(Dense(nClasses))
            model.add(Activation("softmax"))
            # Compile the model
            model.compile(
                loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
            )

            return model

        # define models to test:
        self.sklearn_models = {
            "Dummy Classifier Sklearn": DummyClassifier(),  # Dummy Classifier
            # "Elastic Net": make_pipeline(
            #     RobustScaler(),  # Elastic Net model(Regularized model)
            #     ElasticNet(alpha=0.0005, l1_ratio=0.9),
            # ),
            # "Kernel Ridge": KernelRidge(),  # Kernel Ridge model(Regularized model)
            # # "Bayesian Ridge": BayesianRidge(
            # #     compute_score=True,  # Bayesian Ridge model
            # #     fit_intercept=True,
            # #     n_iter=200,
            # #     normalize=False,
            # # ),
            # "Lasso": make_pipeline(
            #     RobustScaler(),
            #     Lasso(
            #         alpha=0.0005, random_state=2021  # Lasso model(Regularized model)
            #     ),
            # ),
            # "Lasso Lars Ic": LassoLarsIC(
            #     criterion="aic",  # LassoLars IC model
            #     fit_intercept=True,
            #     max_iter=200,
            #     normalize=True,
            #     precompute="auto",
            #     verbose=False,
            # ),
            # "Random Forest": RandomForestClassifier(
            #     n_estimators=300
            # ),  # Random Forest model
            # # "Svm": SVR(),  # Support Vector Machines
            # "Xgboost": XGBClassifier(
            # objective="multi:softprob", verbosity=3
            # ),  # XGBoost model
            # "Gradient Boosting": make_pipeline(
            #     StandardScaler(),
            #     GradientBoostingClassifier(
            #         n_estimators=1,  # GradientBoosting model
            #         learning_rate=0.15,
            #         max_depth=1,
            #         max_features="sqrt",
            #         min_samples_leaf=15,
            #         min_samples_split=10,
            #         # loss="log_loss",
            #         random_state=2021,
            #         verbose=3,
            #     ),
            # ),
        }

        # keras models
        self.keras_models = {
            "Dummy Classifier Keras": KerasClassifier(
                build_dummy_model,
                epochs=10,
                batch_size=32,
                verbose=0,
            ),
            "Neural Network": KerasClassifier(
                build_model_graph,
                epochs=10,
                batch_size=32,
                verbose=0,
            ),
        }

    def model_type(self, model):
        if model in self.sklearn_models.keys():
            return "sklearn"
        elif model in self.keras_models.keys():
            return "keras"
        else:
            return "unknown"

    def init_ml_classifiers(self, algorithms):
        if algorithms.lower() == "all":
            for model in self.sklearn_models.keys():
                self.reg_models[model.title()] = self.sklearn_models[model.title()]
                print(model.title(), (20 - len(str(model))) * "=", ">", "Initialized")
            for model in self.keras_models.keys():
                self.reg_models[model.title()] = self.keras_models[model.title()]
                print(model.title(), (20 - len(str(model))) * "=", ">", "Initialized")
            self.base_models = self.reg_models

        else:
            for model in algorithms:
                if model.lower() in [x.lower() for x in self.base_models.keys()]:
                    print(self.base_models[model])
                    print(
                        model.title(), (20 - len(str(model))) * "=", ">", "Initialized"
                    )

                else:
                    print(
                        model.title(),
                        (20 - len(str(model))) * "=",
                        ">",
                        "Not Initialized",
                    )
                    print(
                        "# Only (Elastic Net,Kernel Ridge,Lasso,Random Forest,SVM,XGBoost,LGBM,Gradient Boosting,Linear Regression)"
                    )

    def show_available(self):
        print(50 * "=")
        print("You can fit your data with the following models")
        print(50 * "=", "\n")
        for model in [m.title() for m in self.base_models.keys()]:
            print(model)
        print("\n", 50 * "=", "\n")

    def train_test_eval_show_results(self, show=True):
        if not self.reg_models:
            raise TypeError("Add models first before fitting")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train,
            self.y_train,
            test_size=self.test_size,
            random_state=2021,
        )

        # Preprocessing, fitting, making predictions and scoring for every model:
        self.result_data = {
            # "R^2": {"Training": {}, "Testing": {}},
            "F1_score": {"Training": {}, "Testing": {}},
        }

        self.p = self.X_train.shape[1]
        self.X_train_n = self.X_train.shape[0]
        self.test_n = self.X_test.shape[0]

        for name, reg_model in self.reg_models.items():
            # fitting the model
            if self.model_type(name) == "sklearn":
                reg_model = reg_model.fit(self.X_train, self.y_train)
            elif self.model_type(name) == "keras":
                history = reg_model.fit(
                    self.X_train, to_categorical(self.y_train), epochs=3
                )
            # make predictions with train and test datasets
            y_pred_train = reg_model.predict(self.X_train)
            y_pred_test = reg_model.predict(self.X_test)

            # calculate the F1 score for training and testing
            f1_score_train = f1_score(self.y_train, y_pred_train, average="weighted")
            f1_score_test = f1_score(self.y_test, y_pred_test, average="weighted")

            # calculate the R-Squared for training and testing
            # r2_train, r2_test = model.score(self.X_train, self.y_train), model.score(
            #     self.X_test, self.y_test
            # )
            (
                self.result_data["F1_score"]["Training"][name],
                self.result_data["F1_score"]["Testing"][name],
            ) = (f1_score_train, f1_score_test)

            if show:
                print("\n", 25 * "=", "{}".format(name), 25 * "=")
                print(10 * "*", "Training", 23 * "*", "Testing", 10 * "*")
                print(
                    "F1    : ",
                    f1_score_train,
                    " " * (25 - len(str(f1_score_train))),
                    f1_score_test,
                )

    def cv_eval_show_results(self, num_models=4, n_folds=5, show=False):
        # prepare configuration for cross validation test
        # Create two dictionaries to store the results of f1_score
        self.f1_results = {"F1_score": {}, "Mean": {}, "std": {}}

        # create a dictionary contains best f1 score results, then sort it
        adj = self.result_data["F1_score"]["Testing"]
        adj_f1_score_sort = dict(sorted(adj.items(), key=lambda x: x[1], reverse=True))

        # check the number of models to visualize results
        if str(num_models).lower() == "all":
            models_name = {
                i: adj_f1_score_sort[i] for i in list(adj_f1_score_sort.keys())
            }
            print()
            print("Apply Cross-Validation for {} models".format(num_models))
            print()

        else:
            print()
            print(
                "Apply Cross-Validation for {} models have highest Adjusted F1_score value on Testing".format(
                    num_models
                )
            )
            print()

            num_models = min(num_models, len(self.base_models.keys()))
            models_name = {
                i: adj_f1_score_sort[i]
                for i in list(adj_f1_score_sort.keys())[:num_models]
            }

        models_name = dict(
            sorted(models_name.items(), key=lambda x: x[1], reverse=True)
        )

        # create Kfold for the cross-validation
        kfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=2021
        ).get_n_splits(self.X_train)

        scoring = {
            "accuracy": make_scorer(accuracy_score, average="weighted"),
            "precision": make_scorer(precision_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "f1_score": make_scorer(f1_score, average="weighted"),
        }

        for name, _ in models_name.items():
            model = self.base_models[name]
            results = cross_validate(
                model,
                self.X_train,
                self.y_train,
                scoring=scoring,
                cv=kfold,
            )

            # save the f1 score reults
            self.f1_results["F1_score"][name] = results["f1_score"].mean()

            # save the RMSE reults
            # self.rmse_results["RMSE"][name] = rms
            # self.rmse_results["Mean"][name] = rms.mean()
            # self.rmse_results["std"][name] = rms.std()

            print(name, (30 - len(name)) * "=", ">", "is Done!")

        if show:
            return self.f1_results  # , self.rmse_results

    def visualize_results(
        self,
        cv_train_test,
        metrics=["f1"],
        metrics_cv=["f1"],
    ):
        if cv_train_test.lower() == "cv":
            # visualize the results of F1_score CV for each model
            self.f1_cv_results = pd.DataFrame(index=self.f1_results["F1_score"].keys())

            # append the max F1_score for each model to the dataframe
            self.f1_cv_results["Max"] = [
                self.f1_results["F1_score"][m].max()
                for m in self.f1_results["F1_score"].keys()
            ]
            # append the mean of all F1_score for each model to the dataframe
            self.f1_cv_results["Mean"] = [
                self.f1_results["Mean"][m] for m in self.f1_results["Mean"].keys()
            ]
            # append the mean of all R-Squared for each model to the dataframe
            self.f1_cv_results["Mean"] = [
                self.f1_results["Mean"][m] for m in self.f1_results["Mean"].keys()
            ]
            # append the min R-Squared for each model to the dataframe
            self.f1_cv_results["Min"] = [
                self.f1_results["F1-Score"][m].min()
                for m in self.f1_results["F1-Score"].keys()
            ]
            # append the std of all R-Squared for each model to the dataframe
            self.f1_cv_results["std"] = [
                self.f1_results["std"][m] for m in self.f1_results["std"].keys()
            ]

            # # visualize the results of RMSE CV for each model
            # self.rmse_cv_results = pd.DataFrame(index=self.rmse_results["RMSE"].keys())
            # # append the max R-Squared for each model to the dataframe
            # self.rmse_cv_results["Max"] = [
            #     self.rmse_results["RMSE"][m].max()
            #     for m in self.rmse_results["RMSE"].keys()
            # ]
            # # append the mean of all R-Squared for each model to the dataframe
            # self.rmse_cv_results["Mean"] = [
            #     self.rmse_results["Mean"][m] for m in self.rmse_results["Mean"].keys()
            # ]
            # # append the min R-Squared for each model to the dataframe
            # self.rmse_cv_results["Min"] = [
            #     self.rmse_results["RMSE"][m].min()
            #     for m in self.rmse_results["RMSE"].keys()
            # ]
            # # append the std of all R-Squared for each model to the dataframe
            # self.rmse_cv_results["std"] = [
            #     self.rmse_results["std"][m] for m in self.rmse_results["std"].keys()
            # ]

            for parm in metrics_cv:
                if parm.lower() in ["f1"]:
                    self.f1_cv_results = self.f1_cv_results.sort_values(
                        by="Mean", ascending=True
                    )
                    self.f1_cv_results.plot(
                        kind="bar",
                        title="Maximum, Minimun, Mean values and standard deviation <br>For f1 values for each model",
                    )
                    self.scores = pd.DataFrame(self.f1_results["f1"])
                    self.scores.plot(
                        kind="box",
                        title="Box plot for the variation of f1 values for each model",
                    )

                # elif parm.lower() in ["r_squared", "rsquared", "r squared"]:
                #     self.r_2_cv_results = self.r_2_cv_results.sort_values(
                #         by="Mean", ascending=False
                #     )
                #     self.r_2_cv_results.plot(
                #         kind="bar",
                #         title="Max, Min, Mean, and standard deviation <br>For R-Squared values for each model",
                #     )
                #     self.scores = pd.DataFrame(self.f1_results["F1-Score"])
                #     self.scores.plot(
                #         kind="box",
                #         title="Box plot for the variation of R-Squared for each model",
                #     )
                else:
                    print("Not avilable")

        elif cv_train_test.lower() == "train test":
            F1 = pd.DataFrame(self.result_data["F1_score"]).sort_values(
                by="Testing", ascending=False
            )

            for parm in metrics:
                if parm.lower() == "f1":
                    # order the results by testing values
                    fig = px.line(
                        data_frame=F1.reset_index(),
                        x="index",
                        y=["Training", "Testing"],
                        title="F1 Score for training and testing",
                    )
                    fig.show()

                # elif parm.lower() == "adjusted r_squared":
                #     # order the results by testing values
                #     fig = px.line(
                #         data_frame=Adjusted_R_2.reset_index(),
                #         x="index",
                #         y=["Training", "Testing"],
                #         title="Adjusted R-Squared for training and testing",
                #     )
                #     fig.show()

                # elif parm.lower() == "mae":
                #     # order the results by testing values
                #     fig = px.line(
                #         data_frame=MAE.reset_index(),
                #         x="index",
                #         y=["Training", "Testing"],
                #         title="Mean absolute error for training and testing",
                #     )
                #     fig.show()

                # elif parm.lower() == "mse":
                #     # order the results by testing values
                #     fig = px.line(
                #         data_frame=MSE.reset_index(),
                #         x="index",
                #         y=["Training", "Testing"],
                #         title="Mean square error for training and testing",
                #     )
                #     fig.show()

                # elif parm.lower() == "rmse":
                #     # order the results by testing values
                #     fig = px.line(
                #         data_frame=RMSE.reset_index(),
                #         x="index",
                #         y=["Training", "Testing"],
                #         title="Root mean square error for training and testing",
                #     )
                #     fig.show()

                else:
                    print("Only (F1)")

        else:
            raise TypeError("Only (CV , Train Test)")

    # def fit_best_model(self):
    #     self.models = list(self.f1_results["Mean"].keys())
    #     self.f1_results_vals = np.array([r for _, r in self.f1_results["Mean"].items()])
    #     self.rmse_results_vals = np.array(
    #         [r for _, r in self.rmse_results["Mean"].items()]
    #     )
    #     self.best_model_name = self.models[
    #         np.argmax(self.f1_results_vals - self.rmse_results_vals)
    #     ]
    #     print()
    #     print(30 * "=")
    #     print("The best model is ====> ", self.best_model_name)
    #     print("It has the highest (R-Squared) and the lowest (Root Mean Square Erorr)")
    #     print(30 * "=")
    #     print()
    #     self.best_model = self.base_models[self.best_model_name]
    #     self.best_model.fit(self.X_train, self.y_train)
    #     print(self.best_model_name, " is fitted to the data!")
    #     print()
    #     print(30 * "=")
    #     self.y_pred = self.best_model.predict(self.test)
    #     self.y_pred = np.expm1(self.y_pred)  # using expm1 (The inverse of log1p)
    #     self.temp = pd.DataFrame({"Id": self.testID, "SalePrice": self.y_pred})

    def show_predictions(self):
        return self.temp

    def save_predictions(self, file_name):
        self.temp.to_csv("{}.csv".format(file_name))
