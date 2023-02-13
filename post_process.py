# Post process the png spectrograms by removing cropping the image to just the spectrogram and
# saving it into the final_spectrograms folder
import os

from PIL import Image

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
        # open the file
        img = Image.open("spectrograms/" + folder + "/" + file)
        # crop the image
        img = img.crop((175, 60, 1040, 445))
        # ensure the output folder exists
        if not os.path.exists(outputfolder + "/" + folder):
            os.makedirs(outputfolder + "/" + folder)
        # save the image
        # reduce the number of colors in the image to 256
        img = img.quantize(256)
        img.save(outputfolder + "/" + folder + "/" + file)
        # print out the file name
        print(folder + "/" + file)

        # use grayscale
        # img = img.convert('L')
