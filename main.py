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
"""glob comes with python installed, so you don't need to install it specifically using pip install or in the python interpreter tab"""
from glob import glob
print("successful")
#preprocessing
"""snoring_files = glob("SnoringNoises/*.wav")
snoring_file_name = ".\Datasets\Snoring Dataset\SnoringNoises\1_18.wav"
snore = wavfile.read(snoring_file_name)
print(snore)"""