import os
import numpy as np
import tensorflow as tf


fileCount = 0
classesCount = 0

for folder in os.listdir("final_spectrograms"):
    # skip any file that is not a folder
    if not os.path.isdir("final_spectrograms/" + folder):
        continue
    classesCount += 1
    for file in os.listdir("final_spectrograms/" + folder):
        # skip any file that is not a png
        if not file.endswith(".png"):
            continue
        fileCount += 1

print ("file count: ", fileCount)

image_size = (865, 385)
batch_size = 2

train_ds = tf.keras.utils.image_dataset_from_directory(
    "final_spectrograms",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "final_spectrograms",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
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

def entireLen():
    return fileCount

# #ratios
# #https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
# trainingLen = round(entireLen()*.7)
# validationLen = round(entireLen()*.15)
# testingLen = round(entireLen()*.15)
#
# # print out training, validation and testing
# print ("training: ", trainingLen)
# print ("validation: ", validationLen)
# print ("testing: ", testingLen)