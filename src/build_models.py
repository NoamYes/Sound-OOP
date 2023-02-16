# define build models functions

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


def build_dummy_model(input_shape, nClasses):
    Input(input_shape)
    model = Sequential()
    model.add(Dense(nClasses))
    # Compile the model
    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    return model


def build_model_graph(input_shape, nClasses):
    Input(input_shape)
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


def build_2d_conv_model(input_shape, nClasses):
    inp = Input(input_shape + (1,))
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
