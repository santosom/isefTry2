import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from PIL import Image

def create_spectrogram(filepath):
    y, sr = librosa.load(filepath)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Power spectrogram')
    # plt.tight_layout()
    plt.axis('off')
    plt.legend().remove()
    return plt


def post_process_spectrogram(file, target):
    img = Image.open(file)
    # crop the image
    img = img.crop((175, 60, 1040, 445))
    # save the image
    # reduce the number of colors in the image to 256
    img = img.quantize(256)
    # use grayscale
    img = img.convert('L')
    img.save(target)
    print ("Saved " + target)


class result:
    def __init__(self, name, score):
        self.name = name
        self.score = score

