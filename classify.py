import os
import shutil
import sys
import lib
import tensorflow
from tensorflow import keras
import numpy as np

# Steps
# * Save the audio file
# * Make a spectrogram
# * Post-process the spectrogram to grayscale, reduced colors and cropped
# * Classify with the the saved model**

tmpAudioFile = sys.argv[1]

# get the filename after the last /
filename_part = tmpAudioFile.split('/')[-1]
tmpAudioFile = 'tmp/' + filename_part
# copy the file to the tmp folder
print("Copied " + sys.argv[1] + " to " + tmpAudioFile)
shutil.copyfileobj(open(sys.argv[1], 'rb'), open(tmpAudioFile, 'wb'))

# generate the spectrogram
# save the spectrogram
plt = lib.create_spectrogram(tmpAudioFile)
spectrogram_file = tmpAudioFile
spectrogram_file = spectrogram_file.split('.')[0]
spectrogram_file = spectrogram_file + '_1_spectrogram.png'
plt.savefig(spectrogram_file)
print("Wrote image to " + spectrogram_file)

# post-process the spectrogram to grayscale, reduced colors and cropped
post_processed_spectrogram = spectrogram_file
post_processed_spectrogram = post_processed_spectrogram.replace('_1_spectrogram.png', '_2_post_processed_spectrogram.png')
lib.post_process_spectrogram(spectrogram_file, post_processed_spectrogram)

# load the keras model in "results/model.h5"
reconstructed_model = keras.models.load_model('results/model.h5')
print("Loaded model from disk")
print(reconstructed_model.summary())
# list all folders in the "spectrograms" folder
labels = os.listdir('audio')
# sort the labels alphabetically
labels.sort()

img = tensorflow.io.read_file(post_processed_spectrogram)
img = tensorflow.image.decode_png(img, channels=1)
img.set_shape([None, None, 1])
img = tensorflow.image.resize(img, (865, 385))
# sess = keras.backend.clear_session
# img = img.eval(session=sess) # convert to numpy array
img = np.expand_dims(img, 0) # make 'batch' of 1

pred = reconstructed_model.predict(img)

lines = []
for i in range(len(pred[0])):
    lines = lines + [str(pred[0][i]) + ' : ' + labels[i]]

lines.sort(reverse=True)
for line in lines:
    print(line)