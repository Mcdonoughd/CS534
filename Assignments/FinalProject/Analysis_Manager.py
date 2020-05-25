'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Analysis_Manager.py Handles the CommandLine choices of ML on the individual cells

'''

import shutil
import os
import Main
import random
import numpy as np
import Logistic_Regression

class AnalysisManager:
    def __init__(self, classDict):
        self.classDict = classDict

    # ask user hwat method of analysis should be done
    def Determine_Analysis_Method(self):
        print("Please choose a Classification Method: \n")
        print("\t 0. Run all Classifiers  ")
        print("\t 1. Logistic Regression ")
        print("\t 2. SVM ")
        print("\t 3. K-means")
        inp = input()
        if inp == "0":
            print("Starting Logistic Regression... ")

            print("Starting SVM... ")

            print("Starting K-means")

        elif inp == "1":
            print("Starting Logistic Regression... ")
        elif inp == "2":
            print("Starting SVM... ")
        elif inp == "3":
            print("Starting K-means")
        else:
            print("Error. Could not read the input. Try again... \n\n")
            self.Determine_Analysis_Method()

    # Organizing Training and testing data
    def OrganizeTrainingData(self):

        training_labels = []
        training_data = []

        testing_data = []
        testing_labels = []
        # TODO REWORK TO LOAD CSV FILE
        for key in self.classDict.keys():

            data = os.listdir(self.classDict[key])

            data = random.shuffle(data)

            datasize = len(data)

            # Obtain Training Data
            training_size = int(datasize*0.2)

            training_data = np.append(training_data,data[:training_size])

            labels = np.full((1, training_size),key)

            training_labels = np.append(training_labels,labels)

            # Obtain Testing Data
            testing_data = np.append(testing_data,data[training_size:])

            labels = np.full((1, datasize-training_size),key)

            testing_labels = np.append(testing_labels,labels)

        # todo make csv

        training_data = np.append(training_data,training_labels,axis=1)
        testing_data = np.append(testing_data,testing_labels,axis=1)
        training_data = np.append(training_data,testing_data)

        training_data.tofile("DATATEST.csv",sep=',')

        return training_data, testing_data, training_labels, testing_labels

#Edge check if directory for image split and storage exists
def makeDirectory():
    # delete current directory and make a new one
    healthy = "./Classified_Cells/healthy"
    unhealthy = "./Classified_Cells/unhealthy"

    if (os.path.isdir(healthy) == True):
        shutil.rmtree(healthy)
    os.mkdir(healthy)

    if (os.path.isdir(unhealthy) == True):
        shutil.rmtree(unhealthy)
    os.mkdir(unhealthy)


# To be used if running from the pipeline
def LoadAnalyzer(dict):
    return AnalysisManager(dict)


# To be used if only running this module alone
def main():
    pipeline = Main.PipelineManager()
    pipeline.DetermineClasses()
    Analyzer = AnalysisManager(pipeline.classDict)
    Analyzer.Determine_Analysis_Method()

if __name__ == '__main__':
   main()

