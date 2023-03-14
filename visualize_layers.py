from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from tensorflow import keras
import tensorflow as tf

# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import numpy as np

# load the model
# model = VGG16()
model = keras.models.load_model('results/model.h5')

# redefine model to output right after the first hidden layer
ixs = [1, 4, 8]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

post_processed_spectrogram = 'test_post_processed_spectrogram.png'
# load the image with the required shape
# img = load_img('test_post_processed_spectrogram.png', target_size=(224, 224))
img = tf.io.read_file(post_processed_spectrogram)
img = tf.image.decode_png(img, channels=1)
img.set_shape([None, None, 1])
img = tf.image.resize(img, (865, 385))
img = np.expand_dims(img, 0) # make 'batch' of 1

# # convert the image to an array
# img = img_to_array(img)
# # expand dimensions so that it represents a single 'sample'
# img = expand_dims(img, axis=0)
# # prepare the image (e.g. scale pixel values for the vgg)
# img = preprocess_input(img)
# # get feature map for first hidden layer

feature_maps = model.predict(img)

print ("feature_maps")

# plot the output from each block
square = 5
for fmap in feature_maps:
 # plot all 25 maps in an 8x8 squares
 ix = 1
 loop = 0
 for _ in range(square):
   for _ in range(square):
     # specify subplot and turn of axis
     ax = pyplot.subplot(square, square, ix)
     ax.set_xticks([])
     ax.set_yticks([])
     # plot filter channel in grayscale
     pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
     # save the figure
     ix += 1
 # pyplot.savefig('feature_map_' + str(loop) + '.png')
 # loop += 1
 # show the figure
 pyplot.show()