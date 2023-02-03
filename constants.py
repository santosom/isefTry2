import tensorflow as tf

#change this back to actual spectrograms
snoringFilePath = 'test_spectrograms/snoring'
ambulanceFilePath = 'test_spectrograms/alarm'
snoringData = tf.data.Dataset.list_files(snoringFilePath)
ambulanceData = tf.data.Dataset.list_files(ambulanceFilePath)

snoringLen = len(snoringData)
ambulanceLen = len(ambulanceData)

def entireLen():
    return snoringLen+ambulanceLen

#ratios
#https://stackoverflow.com/questions/51125266/how-do-i-split-tensorflow-datasets
trainingLen = round(entireLen()*.7)
validationLen = round(entireLen()*.15)
testingLen = round(entireLen()*.15)
