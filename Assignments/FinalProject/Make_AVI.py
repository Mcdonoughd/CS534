'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Make_AVI.py Makes an AVI form a series of multi channel cell images

'''


import os
import shutil
import cv2

# Makes AVI from folder of cell images
def movie(RG_folder):
   image_folder = RG_folder


   print("Please enter the name of the new AVI...")
   video_name = input()
   print("File will be named: " + video_name + ".avi")


   images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
   images.sort()
   frame = cv2.imread(os.path.join(image_folder, images[0]))
   height, width, layers = frame.shape
   video = cv2.VideoWriter(video_name+".avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))

   for image in images:
       print("Reading file: " + image)

       frame = cv2.imread(os.path.join(image_folder, image))

       video.write(frame)

       cv2.imshow('frame', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   video.release()
   cv2.destroyAllWindows()

   print(video_name + ".avi has been created!")

   shutil.move("./"+video_name+".avi", "./Movies/"+video_name+".avi")

   print("It is in the folder ./Movies")

# Deteermine where to read the files from
def determine_Folder():
   print("Please input the location of a folder with cell movie data,")
   print("or press 0 for the default folder")

   inp = input()

   Custom_Folder = "./Cells_to_Detect/Combined_Channels/"

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

   movie(Custom_Folder)


# Used to run directly from this file
if __name__ == '__main__':
   determine_Folder()