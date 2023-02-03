import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
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
print("hello and welcome to the clown show")
#label snoring data by creating another tensor filled entirely with "0", to the proportion of snoringdata, and slapping (zipping) it together
#Currently, snoring's label is 0, ambulance's label is 1 (update as needed)
labelledSnoringData = tf.data.Dataset.zip((snoringData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 0)) ))
labelledAmbulanceData = tf.data.Dataset.zip((ambulanceData, tf.data.Dataset.from_tensor_slices(tf.fill(len(ambulanceData), 1)) ))
allLabelledDatasets = [labelledSnoringData, labelledAmbulanceData]
#congegate all datasets together, note maybe create a for-loop when using more than 2 datasets. this is going to get annoying
ultimateDataset = labelledSnoringData.concatenate(labelledAmbulanceData)
print("dataset type: ", type(ultimateDataset))
#TODO: implement something to split up training and testing sets. I'm not sure if k-fold validation applies here but I'd like to use it somewhere

ultimateDataset.shuffle(3, seed=None, reshuffle_each_iteration=False)
print("dataset shuffled")
ultimateDataset = ultimateDataset.batch(2)
ultimateDataset = ultimateDataset.prefetch(2)
print("dataset batched and prefetched")
train = ultimateDataset.take(constants.trainingLen)
print("part of dataset taken")
test = ultimateDataset.skip(constants.trainingLen).take(constants.testingLen)
print("dataset split into validation, training, and testing")
#make model (woah)
model = Sequential()
#first layer
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
#second layer
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
print("model layers defined")
#TODO: SET A GOOD MINIBATCH SIZE. DO NOT JUST RUN SGD. OH GOSH
optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.1, nesterov=False)
print("model layers optimizer defined")
model.compile(optimizer, loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
print("model successfully complied")
model.fit(
    tf.expand_dims(ultimateDataset, axis=-1),
    y=None,
    batch_size=3,
    epochs=5,
    verbose=1,
    validation_data=test)
print("model fit completed")
model.build(((constants.entireLen()), imageheight, imagewidth, 3))
print(model.summary())
print("model build completed")
#TODO: make sure input shape is correct
print("hello pippin")

