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

from glob import glob

import constants

imageheight, imagewidth = 1400, 500
snoringFilePath = 'spectrograms/snoringNoises'
ambulanceFilePath = 'spectrograms/emergencyNoises'

snoringData = tf.data.Dataset.list_files(snoringFilePath)
ambulanceData = tf.data.Dataset.list_files(ambulanceFilePath)
snoringData = constants.snoringData
ambulanceData = constants.ambulanceData
print("hello")
#label snoring data by creating another tensor filled entirely with "0", to the proportion of snoringdata, and slapping (zipping) it together
#Currently, snoring's label is 0, ambulance's label is 1 (update as needed)
labelledSnoringData= tf.data.Dataset.zip((snoringData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 0)) ))
labelledSnoringData = tf.data.Dataset.zip((snoringData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 0)) ))
labelledAmbulanceData = tf.data.Dataset.zip((ambulanceData, tf.data.Dataset.from_tensor_slices(tf.fill(len(ambulanceData), 1)) ))
allLabelledDatasets = [labelledSnoringData, labelledAmbulanceData]

combined = labelledSnoringData.concatenate(labelledAmbulanceData)
print("dataset type: ", type(combined))

combined.shuffle(3, seed=None, reshuffle_each_iteration=False)
print("dataset shuffled")
combined = combined.batch(2)
combined = combined.prefetch(2)
print("dataset batched and prefetched")
train = combined.take(constants.trainingLen)
print("part of dataset taken")
test = combined.skip(constants.trainingLen).take(constants.testingLen)
print("dataset split into validation, training, and testing")

model = Sequential()

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
print("model layers defined")

print ("train_ns length: ", len(constants.train_ds))
# print the data
print ("train_ns: ", constants.train_ds)

model.build()

print(model.summary())
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
print("model layers optimizer defined")
model.compile(optimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
print("model successfully complied")
model.fit(
    combined,
    y=None,
    batch_size=3,
    epochs=5,
    verbose=1,
    validation_data=test)
print("model fit completed")

print(model.summary())
print("model build completed")
#TODO: make sure input shape is correct
print("hello pippin")