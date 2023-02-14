import os

for folder in os.listdir("final_spectrograms"):
    # skip any file that is not a folder
    if not os.path.isdir("final_spectrograms/" + folder):
        continue
    # get the list of files in the current folder
    files = os.listdir("final_spectrograms/" + folder)
    # if there are more than 10 files in the folder
    if len(files) > 5:
        for i in range(len(files) - 5):
            # delete the file
            os.remove("final_spectrograms/" + folder + "/" + files[i])
            print ("deleted: ", "final_spectrograms/" + folder + "/" + files[i])
