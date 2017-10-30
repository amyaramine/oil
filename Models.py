# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Dense, Dropout, MaxPool2D,  Activation, Flatten, Input, Conv2D, MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def CNN_NCouches(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(8, 3, activation="relu", input_shape=(img_rows, img_cols, color_type)))
    model.add(Convolution2D(16, 3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(8, 3, activation="relu", input_shape=(img_rows, img_cols, color_type)))
    model.add(Convolution2D(16, 3, activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile("adadelta", "binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def get_model1(img_rows, img_cols, color_type):
    p_activation = "relu"
    input_1 = Input(shape=(img_rows, img_cols, color_type), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(8, kernel_size=(3, 3), activation=p_activation)(input_1)
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.25)(img_1)
    img_1 = Flatten()(img_1)

    img_concat = (Concatenate()([img_1, (input_2)]))
    dense_layer = Dropout(0.5)(Dense(128, activation=p_activation)(img_concat))
    #dense_layer = Dropout(0.5)(Dense(128, activation=p_activation)(dense_layer))
    output = Dense(1, activation="sigmoid")(dense_layer)

    model = Model([input_1, input_2], output)
    model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
    model.summary()
    return model


def get_model2():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.2)(img_1)
    img_1 = GlobalMaxPooling2D()(img_1)

    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D()(img_2)

    img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))

    dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_ayer)))
    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1, input_2], output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()
    return model




