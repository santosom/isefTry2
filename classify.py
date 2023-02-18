import sys
import lib
from tensorflow import keras
import numpy as np

# Steps
# * Save the audio file
# * Make a spectrogram
# * Post-process the spectrogram to grayscale, reduced colors and cropped
# * Classify with the the saved model**

tmpAudioFile = sys.argv[1]

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

# np.testing.assert_allclose(
#     reconstructed_model.predict(test_input), reconstructed_model.predict(test_input)
# )
#
# # The reconstructed model is already compiled and has retained the optimizer
# # state, so training can resume:
# reconstructed_model.fit(test_input, test_target)