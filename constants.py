import numpy as np
import tensorflow as tf

#change this back to actual spectrograms
snoringFilePath = 'test_spectrograms/snoring/*png'
ambulanceFilePath = 'test_spectrograms/alarm/*png'
snoringData = tf.data.Dataset.list_files(snoringFilePath)
ambulanceData = tf.data.Dataset.list_files(ambulanceFilePath)

image_size = (1400, 500)
batch_size = 2

train_ds = tf.keras.utils.image_dataset_from_directory(
    "test_spectrograms",
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

# print ("test data: ")
# print (train_ds)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     print (labels)
#     print (images)
#     # for i in range(4):
#     #     ax = plt.subplot(3, 3, i + 1)
#     #     plt.imshow(images[i].numpy().astype("uint8"))
#     #     plt.title(int(labels[i]))
#     #     plt.axis("off")
#     #     plt.savefig('all.png')
#

print ("done")

snoringLen = len(snoringData)
ambulanceLen = len(ambulanceData)

def entireLen():
    return snoringLen+ambulanceLen

#ratios
#https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
trainingLen = round(entireLen()*.7)
validationLen = round(entireLen()*.15)
testingLen = round(entireLen()*.15)
