from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
from tensorflow import keras
# load the model
model = keras.models.load_model('results/model.h5')
# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)