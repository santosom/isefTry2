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
from pathlib import Path
from scipy.io import wavfile
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

import constants

image_size = constants.image_size
imageheight, imagewidth = image_size[1], image_size[0]
batch_size = 2

#make sure this program is loading in the dataset, not constants.py
print("hello mushy")

#load in datasets, split into training and validation (70:30)
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "final_spectrograms",
    label_mode="categorical",
    validation_split=0.3,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

# * This was adapted from a post from Sonjoy Das on Nov 29, 2022 to a StackOverflow question here:
# * https://stackoverflow.com/questions/66036271/splitting-a-tensorflow-dataset-into-training-test-and-validation-sets-from-ker
#Split validation dataset into validation and testing dataset
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take((2*val_batches) // 3)
val_ds = val_ds.skip((2*val_batches) // 3)

#make sure dataset values are accurate
print("training stuff defined")
print("class names are ", train_ds.class_names)
print ("x_train shape before messing with stuff:", train_ds)
print (len(train_ds), "train samples")
print (len(val_ds), "test samples")

#define model
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

optimizer = keras.optimizers.SGD(learning_rate=.00001)
score = model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) #change to categorical_crossentropy

#train model
m = model.fit(train_ds, batch_size = 32, epochs = 75, verbose = 1, validation_data = val_ds)

#evaluate FINAL model on testing dataset
score = model.evaluate(test_ds, verbose=0)
print("test loss:", score[0])
print("test accuracy:", score[1])
print("test recall:", score[2])
print("test precision:", score[3])

# create folder for results, including loss and accuracy graphs + trained model
if not os.path.exists("results"):
    os.makedirs("results")


print("model fit completed")

# * This was adapted from a post from Jason Brownlee on June 17, 2016 on Machine Learning Mastery here:
# * https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
plt.figure(1)
plt.plot(m.history['accuracy'])
plt.plot(m.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.savefig("results/accuracy.png")
print(m.history)
plt.figure(2)
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.savefig("results/loss.png")

#save model
model.save("results/model.h5")

#kaya's bunny
print("hello pippin")
