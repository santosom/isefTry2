# loop over the audio files in the "audio" folder
#
# psuedocode
#
# for (files in audio folder) {
#   print filename
# }
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from pydub.playback import play


def createSpectrogram(filename, filepath):
    print('File: ' + filename)
    y, sr = librosa.load(filepath)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Power spectrogram')
    # plt.tight_layout()
    plt.axis('off')
    plt.legend().remove()
    plt.savefig('spectrograms/' + filename + '.png')
    print("Wrote image spectrograms/" + filename + '.png')
    # create the spectrogram as a numpy array
    #spectrogram = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    #np.save('spectrograms/' + filename + '.npy', spectrogram)
    #print("Wrote numpy spectrograms/" + filename + '.npy')

def generateForFiles(path):
    files = os.listdir(path)
    # i = 0
    for file in files:
        # print(file)
        # if i == 2:
        #     break
        # i += 1
        filename = path + '/' + file # get the full path
        filename = filename.split('.')[0] # removing the .wav part
        filename = filename.replace('/', '_') # replacing the slashes with underscores
        filename = filename[6:] # removing the audio_ part
        createSpectrogram(filename, path + '/' + file)

#generateForFiles('audio/emergency_alarms')
# generateForFiles('audio/snoring')
#generateForFiles('audio/Fire')
generateForFiles('audio/pouring_water')
