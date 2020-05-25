'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Preprocessing.py Handles the pre-processing of cell .tif images

'''


import cv2
import os
import os.path
import Detect
import shutil
import Main
import numpy as np
import ND2_to_TIF
import SetUpManager

class Preprocessor:
    def __init__(self, classDict):
        self.classDict = classDict


    def ReadClasses(self):
        for key in self.classDict.keys():
            value = self.classDict[key]

            print(value)
            if value[-4:] == ".nd2":
                print("Detected an ND2 File....")
                print("ND2 conversion is not a feature yet...")
                exit()
                # TODO the ND2 Conversion to TIF Folder HERE
            elif os.path.isdir(value):
                print("Detected Folder...")
                print("Sorting Channels ...")
                self.sortChannels(value)

    # Given a folder, this function sorts channels into folders based on the file names
        # ND2 File Naming Convention:
        # Date_Time_ROWCOL_COLOR_Channel
        # EX: 2018_10_12_T001_A1_0000_RGB_C1.tif
    def sortChannels(self,folder):
        R_folder, G_folder, B_folder = self.makedir(folder) # check if folders exist
        for file in os.listdir(folder):
                if file[-5:] == "1.tif" or file[-6:] == "1.tiff":
                    print("Channel 1 image detected: "+file)
                    shutil.move(folder+"/"+file, R_folder+"/"+file)

                elif file[-5:] == "2.tif" or file[-6:] == "2.tiff":
                    shutil.move(folder + "/" + file, G_folder + "/" + file)

                elif file[-5:] == "3.tif" or file[-6:] == "3.tiff":
                    shutil.move(folder + "/" + file, B_folder + "/" + file)

                else:
                    print("Error non .TIF file found ...\n\n")
        print("Channels have been sorted! \n\n")
        self.AskToMerge() # ask user if they want to merge images

    # makes a directory for each channel
    def makedir(self,folder):
        channel_1 = os.path.join(folder + '/Channel_1')
        if (os.path.isdir(channel_1) == False):
            os.mkdir(channel_1)

        channel_2 = os.path.join(folder + '/Channel_2')
        if (os.path.isdir(channel_2) == False):
            os.mkdir(channel_2)

        channel_3 = os.path.join(folder + '/Channel_3')
        if (os.path.isdir(channel_3) == False):
            os.mkdir(channel_3)

        return channel_1, channel_2, channel_3

    # asks the user if they woul like to merge color channels
    def AskToMerge(self):
        print("Would you like to also merge color channels? Y/N ")
        inp = input()
        if inp.upper() == "Y":
            print("Merging images...")
            print("Function not available yet! ")
            exit()
            #TODO call merge images function ...

        elif inp.upper() == "N":
            print("Will not merge images...")

        else:
            print("Error please enter Y or N")
            self.AskToMerge()

    # TODO UPDATE THIS FUNCTION
    # combine Rand G files
    def combine(self,R_cells,G_cells,R_folder,G_folder,RG_folder,Test_folder):
       for i in R_cells:
          if i in G_cells:
             r = os.path.join(R_folder + '/' + i)
             g = os.path.join(G_folder + '/' + i)
             green = cv2.imread(g)
             red = cv2.imread(r)
             rb_channel, rg_channel, rr_channel = cv2.split(red)
             gb_channel, gg_channel, gr_channel = cv2.split(green)
             vis = cv2.merge((gb_channel, gg_channel,rr_channel))
             cv2.imwrite(os.path.join(RG_folder + '/' + i),vis)

          else:
             src_name = os.path.join(R_folder + '/' + i)
             dst_name = os.path.join(Test_folder + '/' + i)
             shutil.copyfile(src_name, dst_name)


# To be used if running from Main
def LoadPreprocessor(dic):
    return Preprocessor(dic)


# To be used if only running this module alone
def main():
    pipeline = Main.PipelineManager()
    pipeline.DetermineClasses()
    processor = Preprocessor(pipeline.classDict)
    processor.ReadClasses()


# Used to run directly from this file
if __name__ == '__main__':
   main()
