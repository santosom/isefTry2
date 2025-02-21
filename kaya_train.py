import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.io import wavfile
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt


"""glob comes with python installed, so you don't need to install it specifically using pip install or in the python interpreter tab"""
from glob import glob


imageheight, imagewidth = 400, 500
snoringFilePath = 'test_spectrograms/snoringNoises'
ambulanceFilePath = 'test_spectrograms/emergencyNoises'

snoringData = tf.data.Dataset.list_files(snoringFilePath)
ambulanceData = tf.data.Dataset.list_files(ambulanceFilePath)
#label snoring data by creating another tensor filled entirely with "0", to the proportion of snoringdata, and slapping (zipping) it together
#Currently, snoring's label is 0, ambulance's label is 1 (update as needed)
labelledSnoringData= tf.data.Dataset.zip((snoringData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 0)) ))
labelledAmbulanceData = tf.data.Dataset.zip((ambulanceData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 1)) ))
allLabelledDatasets = [labelledSnoringData, labelledAmbulanceData]
#congegate all datasets together, note maybe create a for-loop when using more than 2 datasets. this is going to get annoying
ultimateDataset = labelledSnoringData.concatenate(labelledAmbulanceData)
print("dataset type: ", type(ultimateDataset))
#use k-fold valiadation to split the training set up!
#make model (woah)
model = Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(128, activation='relu'))

# train the model
model.compile(optimizer='sgd', loss='mse')
print("compiled model")
model.fit(ultimateDataset, epochs=10, batch_size=32)
print("fitted model")

# evaluate the model with a sample audiofile

print("hello pippin")
