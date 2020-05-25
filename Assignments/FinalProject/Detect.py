'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Detect.py Handles the detection of cells in an image

'''


# TODO: Replace LoG with more efficient and FASTER cell detection

import Main
from skimage.color import rgb2gray
from skimage.feature import blob_log
from math import sqrt
import cv2
import numpy as np
import math
from time import time
import os
import shutil
import os.path
import csv
from tqdm import tqdm
import pandas as pd
from PIL import Image

class Detector:
    def __init__(self, classDict):
        self.Total_edgecells = 0
        self.Total_fullcells = 0
        self.All_Detected_Cells = 0
        self.Total_Time = 0
        self.i = 0
        self.classDict = classDict

    # For all file locations in the class dictionary: Run the Nucleic search on the folder
    def DetectNuclei(self):
        for key in self.classDict.keys():
            value = self.classDict[key]
            self.makeDirectories(value)
            self.LoG_Folder(value,key)
            # write to CSV FILE
            self.saveBlobsAsCSV(value, key)
            self.clearBlobs(value)

    # Edge check if directory for image split and storage exists
    def makeDirectories(self,value):
        # delete current directory and make a new one
        detected_cells = value+"/blobs"
        if (os.path.isdir(detected_cells) == True):
            shutil.rmtree(detected_cells)
        os.mkdir(detected_cells)

        edgecells = value+"/edgeCells"
        if (os.path.isdir(edgecells) == True):
            shutil.rmtree(edgecells)
        os.mkdir(edgecells)

    # Detect Nuclei using LoG Blob detection
    def LoG(self,image_location, dir, image_name):

        # Get the Original Image
        image = cv2.imread(image_location, 1)

        # force monochrome
        image_gray = rgb2gray(image)

        # Get the Laplace of the Gaussian
        t0 = time()
        blobs_log = blob_log(image_gray, max_sigma=30, min_sigma=11, num_sigma=10, threshold=.01, overlap=0.3)
        # these params where chosen through manual testing

        t1 = (time() - t0)
        print("done in %0.3fs." % t1)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        # a list of Cells
        list_of_cells = 0
        list_of_edge_cells = 0

        # get the max dimentions of an image
        x_max = image.shape[1]
        y_max = image.shape[0]

        # ID of Cells
        id = 1

        # prepare data for iteration
        blobs_list = [blobs_log]
        colors = ['red']
        sequence = zip(blobs_list, colors)

        print("Cropping out cells ...")
        for _, (blobs, color) in enumerate(sequence):
            for blob in blobs:

                y_o, x_o, r = blob  # init blob

                r = int(math.ceil(r))  # radius of the determined blob
                y = int(y_o)  # Column of determined blob
                x = int(x_o)  # row of determined blob

                # make empty array to fit the cell image
                single_cell = np.zeros((2 * r + 5, 2 * r + 5, 3))

                is_edge = False  # boolean value to determine if the found cell is by the edge

                # Get the image data of a single cell
                for i in range(2 * r + 5):
                    for j in range(2 * r + 5):
                        if x + r - i < x_max and y + r - j < y_max and x + r - i >= 0 and y + r - j >= 0:
                            single_cell[i, j] = image[y + r - j, x + r - i]
                        else:
                            is_edge = True

                # filename of a single cell to be saved
                filename = str(image_name)+ "_Cell_" + str(id) + ".tif"

                if is_edge:
                    cv2.imwrite(os.path.join(dir+"/edgeCells", filename), single_cell)
                    list_of_edge_cells += 1
                else:
                    # write the cell to a file
                    cv2.imwrite(os.path.join(dir+"/blobs", filename), single_cell)

                    # copy value to a list
                    list_of_cells += 1

                id += 1

        num_cells = list_of_cells
        num_edgeCells = list_of_edge_cells
        total_cells = num_cells + num_edgeCells

        return num_cells,num_edgeCells,total_cells,t1

    # TODO Crop the corresponding merged cell files

    # def CropRedGreen(self,file,x,y,r,id, i):
    #     image = cv2.imread("./Cells_to_Detect/Combined_Channels/" + file, 1)
    #
    #     # get the max dimentions of an image
    #     x_max = image.shape[1]
    #     y_max = image.shape[0]
    #     is_edge = False
    #     # make empty array to fit the cell image
    #     single_cell = np.zeros((2 * r + 5, 2 * r + 5, 3))
    #
    #     # Get the image data of a single cell
    #     for i in range(2 * r + 5):
    #         for j in range(2 * r + 5):
    #             if x + r - i < x_max and y + r - j < y_max and x + r - i >= 0 and y + r - j >= 0:
    #                 single_cell[i, j] = image[y + r - j, x + r - i]
    #             else:
    #                 is_edge = True
    #
    #     # filename of a single cell to be saved
    #     filename = "Frame_" +str(i) + "_Cell_" + str(id) + ".tif"
    #
    #     if is_edge:
    #         cv2.imwrite(os.path.join(edgecells_c, filename), single_cell)
    #
    #     else:
    #         # write the cell to a file
    #         cv2.imwrite(os.path.join(detected_cells_c, filename), single_cell)

    # Run LoG on all files in the folder


    # Run LoG on the whole folder of TIF images
    def LoG_Folder(self,dir,key):
        print("Running LoG... ")

        self.Total_edgecells = 0
        self.Total_fullcells = 0
        self.All_Detected_Cells = 0
        self.Total_Time = 0
        self.i = 0

        header = ["FileName", "Full Cells Detected", "Edge Cells Detected","Total Cells Detected","Time Taken (sec)"]
        report_Array = np.array(header)

        for image in os.listdir(dir):
            if image[-4:] == ".tif" or image[-5:] == ".tiff":
                filelocation = dir + "/" + image
                print("Running LoG on " + filelocation)
                currFullCells, currEdgeCells, currTotalCells, time_taken = self.LoG(filelocation, dir, image)
                self.i+=1
                self.Total_edgecells += currEdgeCells
                self.Total_fullcells += currFullCells
                self.All_Detected_Cells += currTotalCells
                self.Total_Time += time_taken

                report_Array = np.vstack((report_Array,np.array([filelocation,currFullCells,currEdgeCells,currTotalCells,time_taken])))

        print("Total Edge Cells Detected: " + str(self.Total_edgecells))
        print("Total Full Cells Detected: " + str(self.Total_fullcells))
        print("Total Cells Detected: " + str(self.All_Detected_Cells))
        print("Total Time Taken: %0.3fs." % self.Total_Time)

        report_Array = np.vstack((report_Array,["Over all files",self.Total_fullcells,self.Total_edgecells,self.All_Detected_Cells,self.Total_Time]))
        # Save report to csv
        np.savetxt(dir+"/Detection_Summary_"+key+".csv", report_Array, delimiter=',',fmt="%s")



    # Converts nucleus image data into CSV
    def saveBlobsAsCSV(self,dirc,key):
        dir = dirc+"/blobs"
        healthy_images = []

        healthy_images += [os.path.join(dir, f) for f in tqdm(os.listdir(dir)) if os.path.isfile(os.path.join(dir, f))]

        df = pd.DataFrame(columns=['image', 'class'])

        #todo Rework this to avoi psudeo tag error
        for img in tqdm(healthy_images):
            image = Image.open(img)
            image = image.resize((20, 20))
            # print(image.size)
            R = []
            for x in range(image.size[0]):
                for y in range(image.size[1]):
                    pixel = image.getpixel((x, y))

                    R.append(pixel)

            df = df.append({'image': R, "class": key}, ignore_index=True)

        df.to_csv(dirc+'/Cells_'+key+'.csv')
        return df

    # Remove Blobs and edgecells folders
    def clearBlobs(self,dir):
        print("Would you like to remove the individual nuclei images? Y/N")
        inp = input()
        if inp.upper() == "Y":
            print("Removing Cells...")
            shutil.rmtree(dir+"/blobs")
            shutil.rmtree(dir+"/edgeCells")

        elif inp.upper() == "N":
            print("Saving Blobs and edges in respected folders")
        else:
            print("Error! please input Y or N")
            self.clearBlobs(dir)



# To be used if running from the pipeline
def LoadDetector(dic):
    return Detector(dic)

# To be used if only running this module alone
def main():
    pipeline = Main.PipelineManager()
    pipeline.DetermineClasses()
    detector = Detector(pipeline.classDict)
    detector.DetectNuclei()


# To be used if only running this module alone
if __name__ == '__main__':
   main()