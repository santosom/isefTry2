import wave
import numpy as np
import pyaudio as pyaudio
from scipy.io.wavfile import write
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio
import lib
import tensorflow
from tensorflow import keras
from lib import result

# Load the model
reconstructed_model = keras.models.load_model('results/model.h5')
print("Loaded model from disk")
print(reconstructed_model.summary())

# get the labels from the model
labels = os.listdir('audio')
labels.remove('.DS_Store')
labels.sort()

WAVE_OUTPUT_FILENAME = "test.wav"

# ---
# Plot audio with zoomed in y axis
def plotAudio(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    ax.margins(2, -0.1)
    # plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    # Save the audio file as a wav file
    write(WAVE_OUTPUT_FILENAME, 44100, X)
    # Generate the spectrogram
    plt = lib.create_spectrogram(WAVE_OUTPUT_FILENAME)
    spectrogram_file = WAVE_OUTPUT_FILENAME
    spectrogram_file = spectrogram_file.split('.')[0]
    spectrogram_file = spectrogram_file + '_spectrogram.png'
    plt.savefig(spectrogram_file)
    print("Wrote image to " + spectrogram_file)
    # Post-process the spectrogram to grayscale, reduced colors and cropped
    post_processed_spectrogram = spectrogram_file
    post_processed_spectrogram = post_processed_spectrogram.replace('_spectrogram.png',
                                                                    '_post_processed_spectrogram.png')
    lib.post_process_spectrogram(spectrogram_file, post_processed_spectrogram)
    img = tensorflow.io.read_file(post_processed_spectrogram)
    img = tensorflow.image.decode_png(img, channels=1)
    img.set_shape([None, None, 1])
    img = tensorflow.image.resize(img, (865, 385))
    img = np.expand_dims(img, 0)  # make 'batch' of 1
    pred = reconstructed_model.predict(img)
    lines = []
    for i in range(len(pred[0])):
        lines += [result(labels[i], pred[0][i])]
    new_list = sorted(lines, key=lambda x: x.score, reverse=True)
    for line in new_list:
        print(line.score, line.name)
    exit(0)

# def predictSound(X):
#     clip, index = librosa.effects.trim(X, top_db=20, frame_length=512, hop_length=64) # Empherically select top_db for every sample
#     stfts = np.abs(librosa.stft(clip, n_fft=512, hop_length=256, win_length=512))
#     stfts = np.mean(stfts,axis=1)
#     stfts = minMaxNormalize(stfts)
#     print("Predicting")
#     # result = model.predict(np.array([stfts]))
#     # predictions = [np.argmax(y) for y in result]
#     # print(lb.inverse_transform([predictions[0]])[0])
#     # plotAudio2(clip)
# ---

CHUNKSIZE = 22050  # fixed chunk size
RATE = 44100
CHANNELS=1
FORMAT = pyaudio.paInt16

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

# noise window
print("Generating noise sample (please be quiet)...")
data = stream.read(10000, exception_on_overflow=False)
noise_sample = np.frombuffer(data, dtype=np.float32)
# plotAudio2(noise_sample)
background_noise_threshold = np.mean(np.abs(noise_sample)) * 10
print("Loud threshold", background_noise_threshold)
audio_buffer = []
near = 0

print ("recording...")
IsRecording = False
while (True):
    # Read chunk and load it into numpy array.
    data = stream.read(CHUNKSIZE, exception_on_overflow=False)
    current_window = np.frombuffer(data, dtype=np.float32)

    # Reduce noise real-time
    # current_window = nr.reduce_noise(y=current_window, y_noise=noise_sample, sr=RATE)

    if (IsRecording == False):
        loudness = np.mean(np.abs(current_window)) * 10
        if (loudness > background_noise_threshold):
            print("Heard something, starting to capture")
            IsRecording = True
        else:
            print("Nothing heard, continuing")
            continue
    else:
        if (near < 10):
            audio_buffer = np.concatenate((audio_buffer, current_window))
            near += 1
        else:
            predictSound(np.array(audio_buffer))
            audio_buffer = []
            IsRecording = False
            near = 0

# close stream
stream.stop_stream()
stream.close()
p.terminate()