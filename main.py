import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

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
snoringFilePath = 'spectrograms/snoringNoises'
ambulanceFilePath = 'spectrograms/emergencyNoises'

snoringData = tf.data.Dataset.list_files(snoringFilePath)
ambulanceData = tf.data.Dataset.list_files(ambulanceFilePath)
#label snoring data by creating another tensor filled entirely with "0", to the proportion of snoringdata, and slapping (zipping) it together
#Currently, snoring's label is 0, ambulance's label is 1 (update as needed)
labelledSnoringData= tf.data.Dataset.zip((snoringData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 0)) ))
labelledAmbulanceData = tf.data.Dataset.zip((ambulanceData, tf.data.Dataset.from_tensor_slices(tf.fill(len(snoringData), 1)) ))
allLabelledDatasets = [labelledSnoringData, labelledAmbulanceData]
#congegate all datasets together, note maybe create a for-loop when using more than 2 datasets. this is going to get annoying
ultimateDataset = labelledSnoringData.concatenate(labelledAmbulanceData)

print("hello pippin")
#use k-fold valiadation to split the training set up!
