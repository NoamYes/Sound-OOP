import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

# visualization libraries
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode

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
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    Input,
    Convolution2D,
    BatchNormalization,
    MaxPool2D,
    Flatten,
)
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier

# from scikeras.wrappers import KerasClassifier

pd.options.plotting.backend = "plotly"


class ML:
    def __init__(
        self,
        data,
        X_train,
        X_train_CNN,
        y_train,
        X_test,
        X_test_CNN,
        y_test,
        testID,
        test_size,
        ntrain,
        nClasses=4,
    ):
        print()
        print("Machine Learning object is created")
        print()

        self.data = data
        self.ntrain = ntrain
        self.test_size = test_size
        self.X_train_1D = X_train
        self.X_train = X_train
        self.X_train_CNN = X_train_CNN
        self.X_test_CNN = X_test_CNN
        self.X_test = X_test
        self.y_test = y_test
        self.testID = testID
        self.y_train = y_train[: self.ntrain]

        self.reg_models = {}
        self.final_models = {}

        def build_dummy_model():
            Input(self.X_train_1D.shape[1:3])
            model = Sequential()
            model.add(Dense(nClasses))
            # Compile the model
            model.compile(
                loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
            )

            return model

        def build_model_graph():
            Input(self.X_train_1D.shape[1:3])
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
                loss="categorical_crossentropy",
                metrics=["accuracy", "AUC", "Precision", "categorical_crossentropy"],
                optimizer="adam",
            )

            return model

        def build_2d_conv_model():
            inp = Input(shape=self.X_train_CNN.shape[1:3] + (1,))
            x = Convolution2D(32, (4, 10), padding="same")(inp)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPool2D()(x)

            x = Convolution2D(32, (4, 10), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPool2D()(x)

            x = Convolution2D(32, (4, 10), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPool2D()(x)

            x = Convolution2D(32, (4, 10), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = MaxPool2D()(x)

            x = Flatten()(x)
            x = Dense(64)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            out = Dense(nClasses, activation="softmax")(x)

            model = Model(inputs=inp, outputs=out)
            opt = Adam(learning_rate=0.01)

            model.compile(
                loss="categorical_crossentropy",
                metrics=["accuracy", "AUC", "Precision", "categorical_crossentropy"],
                optimizer=opt,
            )
            return model

        # define models to test:
        self.sklearn = {
            "X_train": self.X_train_1D,
            "models": {
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
                #     max_depth=5,
                #     n_estimators=300,
                #     nthread=-1,
                #     learning_rate=0.1,
                #     random_state=2023,
                #     reg_alpha=0.3,
                #     reg_lambda=0.1,
                #     colsample_bytree=0.3,
                #     colsample_bylevel=0.8,
                #     objective="multi:softprob",
                #     verbosity=2,
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
            },
        }

        # keras models
        self.keras = {
            "X_train": self.X_train_1D,
            "models": {
                "Dummy Classifier Keras": KerasClassifier(
                    build_dummy_model,
                    epochs=1,
                    batch_size=32,
                    verbose=0,
                ),
            }
            # "Neural Network1": KerasClassifier(
            #     build_model_graph,
            #     epochs=100,
            #     # batch_size=32,
            #     # verbose=3,
            # ),
        }

        self.cnn = {
            "X_train": self.X_train_CNN,
            "models": {
                "Cnn": KerasClassifier(
                    build_2d_conv_model,
                    epochs=100,
                    # batch_size=32,
                    # verbose=3,
                ),
            },
        }

    def model_type(self, model):
        if model in self.sklearn["models"].keys():
            return "sklearn"
        elif model in self.keras["models"].keys():
            return "keras"
        elif model in self.cnn["models"].keys():
            return "cnn"
        else:
            return "unknown"

    def init_ml_classifiers(self, algorithms):
        for model in self.sklearn["models"].keys():
            self.reg_models[model.title()] = self.sklearn["models"][model.title()]
        for model in self.keras["models"].keys():
            self.reg_models[model.title()] = self.keras["models"][model.title()]
        for model in self.cnn["models"].keys():
            self.reg_models[model.title()] = self.cnn["models"][model.title()]
        if str(algorithms).lower() == "all":
            self.final_models = self.reg_models
        else:
            for model in algorithms:
                if model.lower() in [x.lower() for x in self.reg_models.keys()]:
                    self.final_models[model] = self.reg_models[model]
                    # print(self.final_models[model])
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
        for model in [m.title() for m in self.final_models.keys()]:
            print(model)
        print("\n", 50 * "=", "\n")

    def train_test_eval_show_results(self, show=True):
        if not self.reg_models:
            raise TypeError("Add models first before fitting")

        # Preprocessing, fitting, making predictions and scoring for every model:
        self.result_data = {
            # "R^2": {"Training": {}, "Testing": {}},
            "F1_score": {"Training": {}, "Testing": {}},
        }

        for name, reg_model in self.final_models.items():

            # if the model is from cnn then take cnn train
            if self.model_type(name) == "cnn":
                X_train = self.cnn["X_train"]
            elif self.model_type(name) == "keras":
                X_train = self.keras["X_train"]
            elif self.model_type(name) == "sklearn":
                X_train = self.sklearn["X_train"]
            X_train, X_test, y_train, y_test = train_test_split(
                X_train,
                self.y_train,
                test_size=self.test_size,
                random_state=2021,
            )

            # fitting the model
            if self.model_type(name) == "sklearn":
                reg_model = reg_model.fit(X_train, y_train)
            elif self.model_type(name) == "keras" or self.model_type(name) == "cnn":
                history = reg_model.fit(
                    X_train,
                    to_categorical(y_train),
                    # epochs=25,
                    # batch_size=32,
                    # verbose=3,
                )
            # make predictions with train and test datasets
            y_pred_train = reg_model.predict(X_train)
            y_pred_test = reg_model.predict(X_test)

            # calculate the F1 score for training and testing
            f1_score_train = f1_score(y_train, y_pred_train, average="weighted")
            f1_score_test = f1_score(y_test, y_pred_test, average="weighted")

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

            num_models = min(num_models, len(self.final_models.keys()))
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
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="weighted"),
            "recall": make_scorer(recall_score, average="weighted"),
            "f1_score": make_scorer(f1_score, average="weighted"),
        }

        for name, _ in models_name.items():
            model = self.final_models[name]
            if self.model_type(name) == "cnn":
                X_train = self.cnn["X_train"]
            else:
                X_train = self.X_train
            results = cross_validate(
                model,
                X_train,
                self.y_train,
                scoring=scoring,
                cv=kfold,
            )

            # save the f1 score reults
            self.f1_results["F1_score"][name] = results["test_f1_score"]
            self.f1_results["Mean"][name] = results["test_f1_score"].mean()
            self.f1_results["std"][name] = results["test_f1_score"].std()

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

            # append the min F1_score for each model to the dataframe
            self.f1_cv_results["Min"] = [
                self.f1_results["F1_score"][m].min()
                for m in self.f1_results["F1_score"].keys()
            ]
            # append the std of all F1_score for each model to the dataframe
            self.f1_cv_results["std"] = [
                self.f1_results["std"][m] for m in self.f1_results["std"].keys()
            ]

            for parm in metrics_cv:
                if parm.lower() in ["f1"]:
                    self.f1_cv_results = self.f1_cv_results.sort_values(
                        by="Mean", ascending=True
                    )
                    fig = self.f1_cv_results.plot(
                        kind="bar",
                        title="Maximum, Minimun, Mean values and standard deviation <br>For f1 values for each model",
                    )
                    fig.show()
                    self.scores = pd.DataFrame(self.f1_results["F1_score"])
                    fig = self.scores.plot(
                        kind="box",
                        title="Box plot for the variation of f1 values for each model",
                    )
                    fig.show()

                elif parm.lower() in ["f1_"]:
                    self.f1_cv_results = self.f1_cv_results.sort_values(
                        by="Mean", ascending=False
                    )
                    fig = self.f1_cv_results.plot(
                        kind="bar",
                        title="Max, Min, Mean, and standard deviation <br>For R-Squared values for each model",
                    )
                    fig.show()
                    self.scores = pd.DataFrame(self.f1_results["F1_Score"])
                    fig = self.scores.plot(
                        kind="box",
                        title="Box plot for the variation of R-Squared for each model",
                    )
                    fig.show()
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
                else:
                    print("Only (F1)")

        else:
            raise TypeError("Only (CV , Train Test)")

    def fit_best_model(self, lbl_enc):
        self.models = list(self.f1_results["Mean"].keys())
        self.f1_results_vals = np.array([r for _, r in self.f1_results["Mean"].items()])
        self.best_model_name = self.models[np.argmax(self.f1_results_vals)]
        print()
        print(30 * "=")
        print("The best model is ====> ", self.best_model_name)
        print("It has the highest (R-Squared) and the lowest (Root Mean Square Erorr)")
        print(30 * "=")
        print()
        self.best_model = self.final_models[self.best_model_name]
        # choose X_train and X_test based on the model type
        if self.model_type(self.best_model_name) == "cnn":
            X_train = self.cnn["X_train"]
            X_test = self.X_test_CNN
        else:
            X_train = self.X_train_1D
            X_test = self.X_test
        self.best_model.fit(X_train, self.y_train)
        print(self.best_model_name, " is fitted to the data!")
        print()
        print(30 * "=")
        self.y_pred = self.best_model.predict(X_test)
        self.temp = pd.DataFrame(
            {
                "File name": self.testID,
                "SoundLabel": lbl_enc.inverse_transform(self.y_pred),
            }
        )

    def evaluate_model_test(self):
        # choose X_train and X_test based on the model type and save it to file
        if self.model_type(self.best_model_name) == "cnn":
            X_train = self.cnn["X_train"]
            X_test = self.X_test_CNN
        else:
            X_train = self.X_train_1D
            X_test = self.X_test
        self.y_pred = self.best_model.predict(X_test)
        self.f1 = f1_score(self.y_test, self.y_pred, average="weighted")
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.classification_report = classification_report(
            self.y_test, self.y_pred, output_dict=True
        )
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print("F1 Score: ", self.f1)
        print("Accuracy: ", self.accuracy)
        print("Confusion Matrix: \n", self.confusion_matrix)
        print("Classification Report: \n", self.classification_report)
        print(30 * "=")
        print()

    def ignore(self):
        pass

    def save_models(self, directory):
        # save all of the fitted models to certain directory
        for name, model in self.final_models.items():
            # save model with its name, its type and the date (with minutes) delimited with underscore
            file_name = "{}_{}_{}.sav".format(
                name, self.model_type(name), datetime.now().strftime("%Y-%m-%d_%H-%M")
            )
            pickle.dump(model, open(directory + file_name, "wb"))

    def load_models(self, directory):
        # load all of the fitted models from certain directory
        self.final_models = {}
        for file in os.listdir(directory):
            if file.endswith(".sav"):
                model = pickle.load(open(directory + file, "rb"))
                self.final_models[file.split("_")[0]] = model

    def show_predictions(self):
        return self.temp

    def save_predictions(self, file_name):
        self.temp.to_csv("{}.csv".format(file_name))
