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
from matplotlib import pyplot as plt

"""glob comes with python installed, so you don't need to install it specifically using pip install or in the python interpreter tab"""
from glob import glob

#sample for loading in .wav files
filename = "audio/0_1fixed.wav"
fs, wav = wavfile.read(filename)
print(fs)
plt.plot(wav)
plt.show(block=True)
plt.interactive(False)
print("done :)!")
