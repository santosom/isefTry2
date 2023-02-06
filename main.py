import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from keras.optimizers import SGD

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.io import wavfile
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

import constants

imageheight, imagewidth = 1400, 500
snoringData = constants.snoringData
ambulanceData = constants.ambulanceData

model = Sequential()
#first layer
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1400, 500, 3)))
model.add(keras.layers.Dense(64, activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.BatchNormalization())
# #second layer
# model.add(keras.layers.Dense(128, activation='relu'))
# model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
# model.add(keras.layers.Dropout(0.2))
# model.add(keras.layers.BatchNormalization())
# print("model layers defined")
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
# print("model layers optimizer defined")
model.compile(optimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
print("model successfully complied")

print ("train_ns length: ", len(constants.train_ds))
# print the data
print ("train_ns: ", constants.train_ds)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=constants.image_size + (3,), num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    constants.train_ds,
)
    # x=constants.train_ds,
    # y=None,
    # batch_size=3,
    # epochs=5,
    # verbose=1,
    # validation_data=constants.val_ds)
print("model fit completed")
model.build(((constants.entireLen()), imageheight, imagewidth, 3))
print(model.summary())
print("model build completed")
#TODO: make sure input shape is correct
print("hello pippin")

