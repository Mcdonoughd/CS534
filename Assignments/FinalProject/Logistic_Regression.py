'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Logistic_Regression.py Handles the Machine Learning analysis of individual cells

'''

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
from random import shuffle
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from graphing import graphThings
import cv2
import os

class Logistic_Regression:
    def __init__(self, classDict):
        self.classDict = classDict


IMAGE_SIZE = 20


def processArray(array):
    X = []

    for i in range(len(array)):
        c = array[i]
        for char in '[]\n\r,':
            c = c.replace(char, '')

        result = np.array(c.split(' '), dtype=str)
        result = result[result != '']
        result = list(map(lambda x: float(x), result))
        X.append(result)

    return X


def remakeImage(df, size=IMAGE_SIZE, isCancer=False):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    output = np.zeros((size[0], size[1], 3))
    for x in range(size[0]):
        for y in range(size[1]):
            red_val = int(df['red'][x * size[1] + y])
            green_val = int(df['green'][x * size[1] + y])
            if isCancer:
                image.putpixel((x, y), (0, green_val, 0))
                output[x, y] = (0, green_val, 0)
            else:
                image.putpixel((x, y), (red_val, 0, 0))
                output[x, y] = (red_val, 0, 0)

    return image, output


def remakeColoredImage(array, size=IMAGE_SIZE):
    if type(size) == int:
        size = (size, size)
    image = Image.new('RGB', (size[0], size[1]))
    for x in range(size[0]):
        for y in range(size[1]):
            pixel = array[y, x]
            image.putpixel((x, y), (int(pixel[0]), int(pixel[1]), int(pixel[2])))

    return image


def combineImages(image_arrays, size=IMAGE_SIZE):
    # Width is constant, Height is defined by how many images it can fit.
    #
    # Num per row =
    width = 800
    numRows = int(len(image_arrays) / (width / size)) + 1
    height = numRows * size

    img_arr = np.zeros((height, width, 3))
    # print(img_arr.size)
    x = 0
    y = 0
    x_incr = size
    y_incr = size

    dictionary = image_arrays.to_dict('records')
    for img in dictionary:
        if x >= width:
            x = 0
            y += y_incr
        _, array = remakeImage(img, isCancer=img['class'] == 'cancer')
        img_arr[y:y + size, x:x + size] = array
        x += x_incr

    image = remakeColoredImage(img_arr, size=(width, height))
    return image


def processImages(dir, size=IMAGE_SIZE):
    cancer_images = []
    healthy_images = []

    baseName = dir + "/healthy"
    healthy_images += [join(baseName, f) for f in tqdm(listdir(baseName)) if isfile(join(baseName, f))]
    baseName2 = dir + '/unhealthy'
    cancer_images += [join(baseName2, f) for f in tqdm(listdir(baseName2)) if isfile(join(baseName2, f))]

    df = pd.DataFrame(columns=['red', 'class'])
    both = healthy_images + cancer_images

    shuffle(both)
    for img in tqdm(both):
        image = Image.open(img)
        image = image.resize((size, size))
        # print(image.size)
        R = []
        G = []


        for x in range(image.size[0]):
            for y in range(image.size[1]):
                pixel = image.getpixel((x, y))

                R.append(pixel[0])
                G.append(pixel[1])

        if 'unhealthy' in img:
            df = df.append({'red': R, 'green': G, "class": 'cancer'}, ignore_index=True)
        else:
            df = df.append({'red': R, 'green': G, "class": 'healthy'}, ignore_index=True)
    # TODO Remove CSV
    df.to_csv('data_' + (2 * str(size)) + '.csv')
    return df


def train_test_split(X, where=0.8):
    chop = int(len(X) * where)
    train = X[:chop]

    test = X[chop:]

    return train, test


def loadData(size=IMAGE_SIZE):
    df = pd.read_csv('data_' + (2 * str(size)) + '.csv')

    df['red'] = processArray(df['red'])
    df['green'] = processArray(df['green'])

    return df



def logistic(df):
    train, test = train_test_split(df)
    clf = LogisticRegression()
    clf.fit(list(train['red'].values), train['class'].values)

    score = clf.score(list(test['red'].values), test['class'].values)
    print("Logistic Score: ", score)

    prediction = clf.predict(list(test['red'].values))

    print(prediction)

    # TODO Make a scatterplot of all the images in relation to each other

    healthy = test[prediction == 'healthy']
    cancer = test[prediction == 'cancer']


    # TODO MAKE THIS A SEPERATE THING
    # TODO make histograms of the cell images by color frequency
    # TODO make histograms of the cell images by color location in a 1d vector
    # TODO MAKE collage of cells in a vector!

    graphThings([healthy, cancer], ratio=False)

    result = combineImages(healthy)
    result.save("healthy_log" + str(IMAGE_SIZE) + ".jpg")
    result = combineImages(cancer)
    result.save("cancer_log" + str(IMAGE_SIZE) + ".jpg")

    print("Logistic regression had an estimated accuracy of :"+str(score))

    return clf


def Analyze(dir):
    print("Loading Cells from: " + dir)

    print("Giving images Labels ... ")
    GiveLabels(dir)


    df = processImages("./Classified_Cells")
    # df = loadData()

    print("Started Training...")

    logistic(df)

    print("Finished")

# Give all cells a label that are in a folder
def GiveLabels(dir):
    images = os.listdir(dir)
    for image in images:
        readCell(image)

# given a file name calc the max illuminate pixel in the image
def readCell(filename):
    print("Reading file: "+filename)
    img = cv2.imread(os.path.join("./Detected_Cells/blobs_MultiChannel/", filename), 1)  # load the img as just BGR COLOR SPACE
    avg = np.mean(img[:, :, 1])
    max = np.max(img[:, :, 1])
    threshold = avg + (.5 * avg)

    # Determine if Cells are healthy or not by the expression of protein
    if (max < threshold):
        cv2.imwrite(os.path.join('./Classified_Cells/healthy/' + filename), img)

        return 0
    else:
        # unhealthy case
        cv2.imwrite(os.path.join('./Classified_Cells/unhealthy/' + filename), img)

        return 1


def determine_Folder():
    makeDirectory()
    print("Please input the location of a file or folder with cell movie data,")
    print("or press 0 for the default folder")

    inp = input()

    Custom_Folder = "./Detected_Cells/blobs_MultiChannel"


    # If Default
    if (inp == "0"):
        print("Using default folder ...")

    # If folder detected
    elif (os.path.isdir(inp)):
        print("Folder detected ... ")
        print("Using Folder " + inp)
        Custom_Folder = inp

    # If folder detected
    elif (os.path.isfile(inp)):
        print("File detected ... ")
        print("Using File " + inp)
        Custom_Folder = inp

    else:
        print("Error! Cannot read given option. Make sure that the file or folder exists and try again. \n\n")
        determine_Folder()

    Analyze(Custom_Folder)


# To be used if running from the pipeline
def LoadClassifier(dict):
    return Logistic_Regression(dict)


# To be used if only running this module alone
def main():
    determine_Folder()

# To be used if only running this module alone
if __name__ == '__main__':
    main()
