'''

CIRA 2.0 - by Daniel McDonough 7/1/19
ND2_to_TIF.py Converts ND2 Nixon camera files to TIF frame images

'''


# All current ND2Reader packages do not work with Z stack of 0 therefore cannot be done with our dataset.
# TODO Use Jython to have ImageJ run the ND2 files using the Bio-formats plugin
# (Warning: this is known to take up a lot of ram)


import os
import scipy.misc
from nd2reader import ND2Reader
from nd2reader.parser import Parser
import matplotlib.pyplot as plt
# from pims import ND2_Reader
# from pims import

#convert a whole folder of ND2 images to TIF
def Convert_Folder(folder):
    for file in os.listdir(folder):
        # TODO Check if ND2
        Convert_File(file,folder)

def Convert_File(File, directory = None):
    print("Converting Movie: " + str(File))
    if directory == None:
        frames = ND2Reader(File)
    else:
        frames = ND2Reader(str(directory)+"/"+str(File))

    parser = frames.parser
    print(parser.get_image(0))
    image = frames.get_frame_2D(c=0, t=0, z=0, x=2560, y=2160, v=4)
    print(frames.metadata)
    plt.imshow(image)
    print(image)
    scipy.misc.toimage(image, cmin=0.0, cmax=255).save(
        "./To_Preprocess/Frame_" + str(0) + "_" + str(123) + ".jpg")

    # with ND2_Reader(str(directory)+"/"+str(File)) as frames:
    #
    #     # frames.bundle_axes = 'zyx'  # when 'z' is available, this will be default
    #     # frames.default_coords['c'] = 1  # 0 is the default setting
    #     for frame in frames[:3]:
    #         print(frame)
    #         scipy.misc.toimage(frame, cmin=0.0, cmax=255).save(
    #             "./To_Preprocess/Frame_" + str(0) + "_" + str(123) + ".jpg")
    #         input("Press Enter to continue...")
    # do something with 3D frames in channel 1
    # with frames as images:
    #     # width and height of the image
    #     print(images.metadata)

    # print(images.frame_shape)
    # print(images.colors)
    # print(images.sizes)
    # input("Press Enter to continue...")
    # scipy.misc.toimage(images, cmin=0.0, cmax=255).save(
    #     "./To_Preprocess/Frame_" + str(0) + "_" + str(File) + ".jpg")

    # i = 0
    # File = File[:-4]  # Remove the .ND2 extension
    # print(frames.frame_shape)
    #
    # for frame in frames:
    #     print(frame)
    #     plt.imshow(frame)
    #
    #     input("Press Enter to continue...")
    #     scipy.misc.toimage(frame, cmin=0.0, cmax=255).save(
    #         "./To_Preprocess/Frame_" + str(i) + "_" + str(File) + ".jpg")
    #     # do something with 3D frames in channel 1
    #     print(frames[i])
    #
    #     input("Press Enter to continue...")
    #     # scipy.misc.toimage(frame, cmin=0.0, cmax=255).save("./To_Preprocess/Frame_"+str(i)+"_"+str(File)+".jpg")
    #     i +=1
    # frames.close()

def determine_Folder():
    print("Please input the location of a file or folder with cell movie data,")
    print("or press 0 for the default folder")

    inp = input()
    Custom_Folder = "./ND2"

    # If Default
    if (inp == "0"):
        print("Using default folder ...")
        Convert_Folder(Custom_Folder)

    # If folder detected
    elif (os.path.isdir(inp)):
        print("Folder detected ... ")
        print("Using Folder " + inp)
        Convert_Folder(inp)

    # If folder detected
    elif (os.path.isfile(inp)):
        print("File detected ... ")
        print("Using File " + inp)
        Convert_File(inp)

    else:
        print("Error! Cannot read given option. Make sure that the file or folder exists and try again. \n\n")
        determine_Folder()
        exit()




if __name__ == "__main__":
    determine_Folder()