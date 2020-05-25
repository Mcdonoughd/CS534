'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Main.py This handles the initial user interface command line prompts

'''


import os
import os.path
import Preprocessing
import Make_AVI
import SetUpManager
import Detect
import Analysis_Manager
import Make_GIF
import ND2_to_TIF

# Pipeline manager that keeps track of the actions in the pipeline
class PipelineManager:

    def __init__(self):
        self.classDict = dict()
        self.action = 0


    # Min interface the uses to choose what they want to do
    def ChooseTask(self):

        print("Hello, what would you like to do?")
        print("\t 0 Run Pipeline")
        print("\t 1 Pre-process Images")
        print("\t 2 Detect Nuclei")
        print("\t 3 Classify Nuclei")
        print("\t 4 Make GIF from Images")
        print("\t 5 Make AVI from Images")
        # print("\t 6 Test Detection Methods") # TODO make this
        # print("\t 7 Test Classification Methods") # TODO make this


        inp = input()
        if (inp == "0"):
            print("Preprocessing initiated...\n")
            self.action = 0

        elif (inp == "1"):
            print("Preprocessing initiated...\n")
            self.action = 1

        elif(inp == "2"):
            print("Detecting Cells process initiated...\n ")
            self.action = 2
        elif(inp == "3"):
            print("Classifying Cells process initiated...\n ")
            self.action = 3

        elif (inp == "4"):
            print("Making GIF process initiated... \n")
            self.action = 4

        elif (inp == "5"):
            print("Making AVI process initiated... \n")
            self.action = 5

        elif(inp == "6"):
            print("Detection testing process initiated...\n")
            self.action = 6

            # TODO This will be used to test different methods of detection

        elif(inp == "7"):
            print("Classification testing process initiated...\n")
            self.action = 7
            # TODO THis will be used to test different methods of classification

        else:
            print("Error! Given input is not an option!... Restarting... \n\n ")
            self.ChooseTask()

    # Ask the user what features are being tested...
    def DetermineClasses(self):
        print("Please input the name of the testing classes one at a time (Including Wild-Type): \n")
        inp = input()
        featurename = inp
        self.getLocationofFiles(featurename)
        self.checkMoreClasses()

    # get the location of the files of the corresponding features
    def getLocationofFiles(self,featurename):
        print("Please input the location of the class .TIF folder or .ND2 File")
        inp = input()
        if os.path.isdir(inp) or (os.path.isfile(inp) and inp[-4:] == ".nd2") :
            self.classDict[featurename] = inp
        else:
            print("Error! Directory does not exist or File is not ND2! ")
            self.getLocationofFiles(featurename)

    # checks if the user wants to upload more features ...
    def checkMoreClasses(self):
        print("Are there any more classes: Y/N ")
        inp = input()
        if inp.upper() == "Y":
            self.DetermineClasses()
        elif inp.upper() == "N":
            return
        else:
            print("Error please input Y or N")
            self.checkMoreClasses()

    # Run the chosen action
    def runModule(self):
        if self.action == 0:
            self.RunPipeline()
        elif self.action == 1:
            processor = Preprocessing.LoadPreprocessor(self.classDict)
            processor.ReadClasses()
        elif self.action == 2:
            detector = Detect.LoadDetector(self.classDict)
            detector.DetectNuclei()
        elif self.action == 3:
            Analysis_Manager.determine_Folder()
        elif self.action == 4:
           Make_GIF.determine_Folder()
        elif self.action == 5:
            Make_AVI.determine_Folder()

    def RunPipeline(self):
        Preprocessor = Preprocessing.LoadPreprocessor(self.classDict)
        Preprocessor.ReadClasses()
        self.updateDictAfterPreprocessing()
        # update the dictionary to account for the new channel folders
        Detector = Detect.LoadDetector(self.classDict)
        Detector.DetectNuclei()
        # update the dictionary to account for the new individual nuclei folders


        # ask user for the Channel with Nucleus data # TODO remove this with AI feature to detect nuclei in a sample frame

    def getNucleiChannel(self,key):
        print("What channel contains nucleic data for " + key + "? 1/2/3 ")
        inp = input()
        if inp == "1":
            return "/Channel_1"
        elif inp == "2":
            return "/Channel_2"
        elif inp == "3":
            return "/Channel_3"
        else:
            print("Error please enter 1,2, or 3")
            self.getNucleiChannel()

    def updateDictAfterPreprocessing(self):
        for key in self.classDict:
            channel = self.getNucleiChannel(key)
            newvalue = self.classDict[key] + channel
            self.classDict[key] = newvalue

    def updateDictAfterDetection(self):
        for key in self.classDict:
            newvalue = self.classDict[key] + "/Cells_"+key+".csv"
            self.classDict[key] = newvalue

    def updateDict(self, key, newValue):
        self.classDict[key] = newValue

def main():
    SetUpManager.SetUpDirectiory_Main() # Set up all files needed
    pipeline = PipelineManager() # Set up the pipeline
    pipeline.ChooseTask() # Determine the Actions that need to
    pipeline.DetermineClasses() # Determine the features that are being tested ...
    pipeline.runModule() # Run the chosen module



if __name__ == '__main__':
    main()

