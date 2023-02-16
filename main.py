import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, MaxPooling2D
from keras.optimizers import SGD

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.io import wavfile
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

import constants

image_size = constants.image_size
imageheight, imagewidth = image_size[1], image_size[0]
batch_size = 2
print("hello mushy")

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "final_spectrograms",
    label_mode="categorical",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

#testing split from https://stackoverflow.com/questions/66036271/splitting-a-tensorflow-dataset-into-training-test-and-validation-sets-from-ker
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take((2*val_batches) // 3)
val_ds = val_ds.skip((2*val_batches) // 3)

# for images, labels in train_ds.take(1):
#     print (labels)
#     print (images)

print("training stuff defined")
print("class names are ", train_ds.class_names)
print ("x_train shape before messing with stuff:", train_ds)
print (len(train_ds), "train samples")
print (len(val_ds), "test samples")

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape = (imagewidth, imageheight, 1)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(constants.classesCount,  activation = 'softmax'))
model.summary()
print("model layers defined")
optimizer = keras.optimizers.SGD(learning_rate=.00001)
print("model layers optimizer defined")
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy']) #change to categorical_crossentropy
print("model successfully complied")

m = model.fit(train_ds, batch_size = 32, epochs = 100, verbose = 1, validation_data = val_ds)

# #TODO: make sure input shape is correct
if not os.path.exists("results"):
    os.makedirs("results")

print("model fit completed")
plt.figure(1)
plt.plot(m.history['accuracy'])
plt.plot(m.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.savefig("results/accuracy.png")

# print the m history files
print(m.history)

# summarize history for loss
plt.figure(2)
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
# write the plt to a results folder
# make the results folder if needed
plt.savefig("results/loss.png")

model.save("results/model.h5")

print("hello pippin")