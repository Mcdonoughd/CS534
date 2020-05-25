#SVC by Daniel McDonough
import matplotlib.pyplot as plt
from copy import deepcopy
import random
from sklearn.datasets import load_digits
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import operator
from sklearn.svm import SVC
import re
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False



def fmeasure(con_matrix):

    if con_matrix.shape == (1,1):
        TP = con_matrix[0,0]
        FN = 0
        FP = 0
        TN = 0
    else:
        #note TP and TN are switched here becuase we are testing Notckd
        TP = con_matrix[1,1]
        FN = con_matrix[0,1]

        FP = con_matrix[1,0]

        TN = con_matrix[0,0]

    PRE = TP/(TP+FP)
    REC = TP/(TP+FN)
    if math.isnan(PRE):
        PRE = 0
    if math.isnan(REC):
        REC = 0

    fmeasure = 2*((PRE*REC) / (PRE+REC))

    if math.isnan(fmeasure):
        fmeasure = 0

    return fmeasure



#function that cleans empty columns in the dataset and major missing columns
def resize_data(data):

    print("The current shape: ",data.shape)
    #columns_titles = data[0,:] #save the first row
    data = np.delete(data, 0, 0) #delete first row
    #print(columns_titles)

    nullcount_row = [0]*len(data[:,0]) #number of columns
    nullcount_col = [0]*len(data[1,:]) #number of rows

    #remove bad data entries
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            data[i,j] = re.sub('\s+', '', data[i,j]) #remove special char if that exist
            if data[i,j] == "NA" or data[i,j] == "":
                nullcount_row[i] += 1 #get a count of NA features per data entry

    shaped_data = []
    empty_thresh = len(data[0,:])*0.25 #threshold of empty datapoint to remove the entry

    #if a data entry is missing more than 25% of features then remove it...
    for k in range(len(nullcount_row)):
        if not nullcount_row[k] >= empty_thresh:
            shaped_data.append(data[k]) #append good data
        else:
            print("Removed Entry: ", k)
    shaped_data = np.array(shaped_data)
    #print(shaped_data.shape)
    #print(shaped_data)


    #remove bad features
    for j in range(len(shaped_data[0,:])):
        for i in range(len(shaped_data[:,0])):
            if shaped_data[i,j] == "NA" or shaped_data[i,j] == "":
                nullcount_col[j] += 1 #get a count of NA features per data entry

    clean_data = []
    #print(type(clean_data))
    empty_thresh = len(data[:,0])*0.25 #threshold of empty datapoints to remove the feature
    #print(empty_thresh)

    #if a feature is missing more than 25% of data then remove it...
    for k in range(len(nullcount_col)):
        if not nullcount_col[k] >= empty_thresh:
            clean_data.append(shaped_data[:,k]) #append good data
        else:
            print("Removed Feature: ", k)
    clean_data = np.array(clean_data).T
    #print(clean_data.shape)
    lastcol = clean_data.shape[1]-1
    #print("Last column: ",lastcol)
    labels = clean_data[:,lastcol]#get the labels
    clean_data = np.delete(clean_data, lastcol, 1) #delete the labels

    #print(clean_data)

    return clean_data,labels



#function that gets the mean value dispite the type of data
def getmean(good_data):
    #avg = 0
    good_data = list(map(float, good_data)) #turn strings into floats
    total = sum(good_data)
    avg = total / len(good_data)
    #print(good_data)
    #print(good_data.shape)

    return avg

#function that returns the avg and list of missing data points
def mean_missing(data,col):

    good_data = [] #used to calculate the sum / average
    bad_data = [] #list of locations to data points with NA
    for i in range(data.shape[0]):
        if data[i, col] == '' or data[i, col] == "NA":
            bad_data.append((i, col))
        else:
            good_data.append(data[i, col])
    #print(col)
    mean = getmean(good_data)
    return mean,bad_data

def discrete_to_num(data,col):
    good_data = []  # used to calculate the sum / average
    bad_data = [] #list of locations to data points with NA
    #get the good data in the column
    for i in range(data.shape[0]):
        if data[i, col] == '' or data[i, col] == "NA":
            bad_data.append((i, col))
            good_data.append(data[i, col])
        else:
            good_data.append(data[i, col])

    unique = list(set(good_data)) # set of unique values
    unique.remove("NA")
    val = list(range(len(unique))) #values corresponding to each unique value
    #print(unique)
    # print(val)
    #convert good_data into ints
    for j in range(len(good_data)):
        for k in range(len(unique)):
            if good_data[j] == unique[k]:
                good_data[j] = val[k]
                break;
    unique = good_data

    unique= list(filter(lambda a: a != "NA", unique))

    #print(val)
    # print(col)
    # print(good_data)
    mean = getmean(unique)
    return mean, bad_data,good_data

#function the replaces missing data with
def calcdata(data):
    for i in range(data.shape[1]):

        if isfloat(data[5,i]): #alter the numerical data
            avg, list = mean_missing(data,i)
            for j in range(len(list)): #replace NA
                xy = list[j]
                data[xy[0], xy[1]] = avg
        else:
            #change discrete values to numbers
            avg, list,data[:,i] = discrete_to_num(data,i)
            for j in range(len(list)): #replace NA
                xy = list[j]
                data[xy[0], xy[1]] = avg

    return data


def cleanupdata(data):
    shaped_data,labels = resize_data(data) #delete bad data
    print("The New shape: ", shaped_data.shape)
    avg_data = calcdata(shaped_data)
    print("Missing Values filled with Averages")
    avg_data = np.delete(avg_data, 0, 0) #delete first row (aka feature names)
    #print(avg_data)
    return avg_data,labels


def main():


    fileobject = np.loadtxt("chronic_kidney_disease_full.csv", delimiter=",",dtype='str')


    data,labels = cleanupdata(fileobject)
    #print(data)
    training_size = int(math.floor(len(data[:,0])*0.8)) # 80% training size
    #print(training_size)

    training = data[:training_size] #training data
    validation = data[training_size:] #vaildation data

    training_labels = labels[:training_size]
    true_labels = labels[training_size+1:]

    # print(len(validation))
    # print(len(true_labels))
    clf = SVC(kernel='linear').fit(training,training_labels)
    #print("Accuracy: ",clf.score(training, training_labels))
    pred = clf.predict(validation)
    con_matrix = confusion_matrix(true_labels, pred)
   # print("SVC Linear Confusion Matrix: \n",con_matrix)
    #print("Linear Kernel F1 Score: ",f1_score(true_labels, pred))
    print("\nSVC Linear F-Measure",fmeasure(con_matrix))


    rbf = SVC(kernel='rbf',gamma="scale").fit(training, training_labels)
    #print(rbf.score(training, training_labels))
    pred = rbf.predict(validation)
    con_matrix = confusion_matrix(true_labels, pred)
    #print("SVC RBF Confusion Matrix: \n",con_matrix)
   # print("RBF Kernal F1 Score: ", f1_score(true_labels, pred))
    print("SVC RBG F-Measure",fmeasure(con_matrix))



    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(training,training_labels)
    #print(rfc.score(training, training_labels))
    pred = rfc.predict(validation)
   # print("Random Forest Classifier F1 Score: ", f1_score(true_labels, pred))
    con_matrix = confusion_matrix(true_labels, pred)
    #print("Random Forest Classifier Confusion Matrix: \n",con_matrix)
    print("Random Forest Classifier F-Measure",fmeasure(con_matrix))


if __name__ == "__main__":
    main()