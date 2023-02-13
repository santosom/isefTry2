import os

main = "/Users/kaya/Downloads/ESC-50-master/"
csvFile = main + "meta/esc50.csv"
outputBase = "audio/"
wantedCategories = ["pouring_water", "laughing", "clapping", "crying_baby", "foot_steps", "door_creak", "door_knock", "alarm", "glass_breaking", "car_alarm", "engine", "dog"]

# open the csv file
with open(csvFile, "r") as f:
    # read the lines in the file
    lines = f.readlines()
    # loop over the lines
    for line in lines:
        # split the line into parts
        parts = line.split(",")
        # get the filename
        filename = parts[0]
        # get the category
        category = parts[3]
        # if the category is not in the list of categories, skip it
        if category not in wantedCategories:
            continue
        # get the fold
        fold = parts[5]
        # skip the header
        if filename == "filename":
            continue
        # skip the test files
        if fold == "5":
            continue
        # print out the filename and category
        print(filename, category)
        # create the output folder
        outputFolder = outputBase + category + "/"
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        # copy the file
        os.system("cp " + main + "audio/" + filename + " " + outputFolder + filename)
