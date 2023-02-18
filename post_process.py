# Post process the png spectrograms by removing cropping the image to just the spectrogram and
# saving it into the final_spectrograms folder
import os

import lib

outputfolder = "final_spectrograms"

# loop over the folders in spectrograms
for folder in os.listdir("spectrograms"):
    # loop over the files in the folder
    # skip any file that is not a folder
    if not os.path.isdir("spectrograms/" + folder):
        continue
    for file in os.listdir("spectrograms/" + folder):
        # skip any file that is not a png
        if not file.endswith(".png"):
            continue
        if not os.path.exists(outputfolder + "/" + folder):
            os.makedirs(outputfolder + "/" + folder)
        lib.post_process_spectrogram("spectrograms/" + folder + "/" + file, outputfolder + "/" + folder + "/" + file)
