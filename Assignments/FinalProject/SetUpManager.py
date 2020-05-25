'''

CIRA 2.0 - by Daniel McDonough 7/1/19
SetUpManager.py This sets up files before a module runs

'''

import os


# This checks if Folders already exist in the directory
def SetUpDirectiory_Main():
   Cells_to_Detect = os.path.join('./Cells_to_Detect')
   if (os.path.isdir(Cells_to_Detect) == False):
      os.mkdir(Cells_to_Detect)

   classify = os.path.join('./Classified_Cells')
   if (os.path.isdir(classify) == False):
      os.mkdir(classify)

   preprocess = os.path.join('./To_Preprocess')
   if (os.path.isdir(preprocess) == False):
      os.mkdir(preprocess)

   gif_folder = os.path.join('./GIF')
   if (os.path.isdir(gif_folder) == False):
      os.mkdir(gif_folder)

   movie_folder = os.path.join('./Movies')
   if (os.path.isdir(movie_folder) == False):
      os.mkdir(movie_folder)

   ND2_folder = os.path.join('./ND2')
   if (os.path.isdir(ND2_folder) == False):
      os.mkdir(ND2_folder)

if __name__ == "__main__":
    SetUpDirectiory_Main()