import tensorflow
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

print("hello pippin")
imageheight, imagewidth = 400, 500
#use k-fold valiadation to split the training set up!
