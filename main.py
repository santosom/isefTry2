import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D, Dense, MaxPooling2D, Input
from keras.optimizers import Adam

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scipy.io import wavfile
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

image_size = (1400, 500)
batch_size = 2
print("hello mushy")
train_ds = tf.keras.utils.image_dataset_from_directory(
    "test_spectrograms",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "test_spectrograms",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

for images, labels in train_ds.take(1):
    print (labels)
    print (images)


print("training stuff defined")
print("class names are ", train_ds.class_names)
print ("x_train shape before messing with stuff:", train_ds)
print (len(train_ds), "train samples")
print (len(val_ds), "test samples")


model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape = (1400, 500, 3)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(2, activation = 'softmax'))
model.summary()
print("model layers defined")
optimizer = keras.optimizers.Adam(learning_rate=0.01)
print("model layers optimizer defined")
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy']) #change to categorical_crossentropy
print("model successfully complied")

model.fit(train_ds, batch_size = 3, epochs = 5, verbose = 1)
# m = model.fit(
#     # constants.train_ds,
#     X_train,
#     Y_train,
#     batch_size=3,
#     epochs=5,
#     verbose=1,
# )

print("model fit completed")
# model.build(((constants.entireLen()), imageheight, imagewidth, 3))
# print("model build completed")
# print("history keys:", m.history.keys())
# #from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# plt.plot(m.history['accuracy'])
# plt.plot(m.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(m.history['loss'])
# plt.plot(m.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# #TODO: make sure input shape is correct
# print("hello pippin")

