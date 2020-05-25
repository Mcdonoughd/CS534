'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Make_GIF.py Makes a GIF form a series of multi channel cell images

'''

import imageio
import os
import cv2

# make gif from a folder of cell images
def makeGIF(dir):

    print("Please enter the name of the new GIF...")
    video_name = input()
    print("File will be named: " + video_name + ".gif")

    folder = os.listdir(dir)
    folder.sort()
    images = []

    print("Making GIF...")
    for filename in folder:
        file_path = os.path.join(dir, filename)
        img = cv2.imread(file_path)
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(RGB_img)

    imageio.mimsave('./GIF/'+video_name+".gif", images, fps=30)

    print("GIF completed!")


# Deteermine where to read the files from
def determine_Folder():
   print("Please input the location of a file or folder with cell movie data,")
   print("or press 0 for the default folder")

   inp = input()

   Custom_Folder = "./Cells_to_Detect/Combined_Channels"

   # If Default
   if (inp == "0"):
      print("Using default folder ...")

   # If folder detected
   elif (os.path.isdir(inp)):
      print("Folder detected ... ")
      print("Using Folder " + inp)
      Custom_Folder = inp

   else:
      print("Error! Cannot read given option. Make sure that the file or folder exists and try again. \n\n")
      determine_Folder()

   makeGIF(Custom_Folder)


# Used to run directly from this file
if __name__ == '__main__':
   determine_Folder()