import matplotlib.pyplot as plt
import librosa
import librosa.display

# this is a nice spectrogram but the y-axis differs per file which isn't good
# it is an implementation of the per-channel energy normalization
def testAmplatudeSpectrogram(filename, file):
    y, sr = librosa.load(file)
    plt.figure(figsize=(14, 5))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # S_DB = librosa.power_to_db(S, ref=np.max)
    # librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time', sr=sr)
    S_pcen = librosa.pcen(S)
    librosa.display.specshow(S_pcen, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('spectrograms/' + filename + '_amp.png')