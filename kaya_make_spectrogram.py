# generate a spectrogram as a function call for a single file
# take a command line argument for the filename
import os
import sys
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import sys

# get a command line argument

# if the argv array is empty exit with an error
if len(sys.argv) < 2:
    print('Please enter a file name')
    sys.exit()

# get a commnad line parameter for the file we want to open
file = sys.argv[1]

# if the file doesn't exist exit with an error
if not os.path.exists(file):
    print('File does not exist')

filename = os.path.basename(file)

# print the file argument and exit
print('File: ' + file)
print('Filename: ' + filename)

# take the file argument and get rid of the ending after the last period
relativePath = file.split('.')[0]
# replace slashes with undercores in the relative path
relativePath = relativePath.replace('/', '_')
print ('Relative Path: ' + relativePath)

# sys.exit(0)

def createAmplatudeSpectrogram(filename, file):
    y, sr = librosa.load(file)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Power spectrogram')
    # hide the legend
    # hide the x axis
    # hide the y axis
    plt.tight_layout()
    plt.savefig('spectrograms/' + filename + '_amp.png')

def createPowerSpectrogram(filename, file):
    y, sr = librosa.load(file)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.power_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Power spectrogram')
    # hide the legend
    # hide the x axis
    # hide the y axis
    plt.tight_layout()
    plt.savefig('spectrograms/' + filename + '_power.png')

# make a spectrogram image of the audio file
# import the necessary packages
createAmplatudeSpectrogram(relativePath, file)
# createPowerSpectrogram(relativePath, file)
