# loop over the audio files in the "audio" folder
#
# psuedocode
#
# for (files in audio folder) {
#   print filename
# }
import os
import lib

def generateForFiles(path):
    files = os.listdir(path)
    for file in files:
        filename = path + '/' + file # get the full path
        filename = filename.split('.')[0] # removing the .wav part
        filename = filename.replace('/', '_') # replacing the slashes with underscores
        filename = filename[6:] # removing the audio_ part
        createSpectrogramAndSaveFile(path + '/' + file)
def createSpectrogramAndSaveFile(filepath):
    plt = lib.create_spectrogram(filepath)
    target = filepath.replace('audio/', 'spectrograms/')
    target = target.split('.')[0]
    target_folder = target[:target.rfind('/')]
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    plt.savefig(target + '.png')
    print("Wrote image to " + target)

# loop through each folder in the audio folder
folders = os.listdir('audio')
for folder in folders:
    print("Processing folder " + folder)
    generateForFiles('audio/' + folder)