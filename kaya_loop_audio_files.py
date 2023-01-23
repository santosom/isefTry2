# loop over the audio files in the "audio" folder
#
# psuedocode
#
# for (files in audio folder) {
#   print filename
# }
import os

# get a list of files in the audio folder
files = os.listdir('audio')

for file in files:
    print(file)
