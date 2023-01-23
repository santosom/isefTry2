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

# get a list of files in the audio folder
files = os.listdir('audio')

def createSpectrogram(filename, filepath):
    y, sr = librosa.load(filepath)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.tight_layout()
    plt.savefig('spectrograms/' + filename + '.png')

for file in files:
    print(file)
    createSpectrogram(file, "audio/"+file)