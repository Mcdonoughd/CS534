'''

CIRA 2.0 - by Daniel McDonough 7/1/19
Detect.py Handles the detection of cells in an image

'''

from math import sqrt
from skimage import filters
from skimage.util import img_as_float
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_laplace
from math import sqrt, log
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from time import time
import os
from scipy import spatial
import shutil
import os.path
import random
from numpy import genfromtxt
import csv

#global defined Strings
cell_dir = 'blobs'
unhealthyDir = 'unhealthyCells'
healthyDir = 'healthyCells'
edgecells = 'edgeCells'
log_masks = 'log_masks'

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def eucldist(p0, p1,xmax,ymax):
    x1 = p0[0]
    y1 = p0[1]
    x2 = p1[0]
    y2 = p1[1]

    dist = (((x2-x1)/xmax)**2)+(((y2-y1)/ymax)**2)
    return math.sqrt(dist)

def AP(listofiou):
    Precision = []
    for i in np.arange(0.0, 1.0, 0.1):
        TP = len(listofiou[np.where(listofiou>=i)])
        FP = len(listofiou[np.where(listofiou<i)])
        deno = TP+FP
        if deno == 0.0:
            Precision.append(0.0)
        else:
            Precision.append(TP/(TP+FP))
        #Precision 5 is equal to the PVOC
    return np.mean(Precision), Precision[5]

def IOU(true,pred):
   true = true / 255
   pred = pred /255
   overlay = true + pred
   intersection  = overlay[np.where(overlay==2)]
   union = overlay[np.where(overlay>=1)]
   IOU = len(intersection)/len(union)
   return IOU


def IOT(true,pred):
   true = true / 255
   pred = pred /255
   overlay = true + pred
   intersection  = overlay[np.where(overlay==2)]
   union = true[np.where(true>=1)]
   IOU = len(intersection)/len(union)
   return IOU

def _compute_disk_overlap(d, r1, r2):
    """
    Compute surface overlap between two disks of radii ``r1`` and ``r2``,
    with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first disk.
    r2 : float
        Radius of the second disk.
    Returns
    -------
    vol: float
        Volume of the overlap between the two disks.
    """

    ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    ratio1 = np.clip(ratio1, -1, 1)
    acos1 = math.acos(ratio1)

    ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    ratio2 = np.clip(ratio2, -1, 1)
    acos2 = math.acos(ratio2)

    a = -d + r2 + r1
    b = d - r2 + r1
    c = d + r2 - r1
    d = d + r2 + r1
    area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d)))
    return area / (math.pi * (min(r1, r2) ** 2))



def _compute_sphere_overlap(d, r1, r2):
    """
    Compute volume overlap between two spheres of radii ``r1`` and ``r2``,
    with centers separated by a distance ``d``.
    Parameters
    ----------
    d : float
        Distance between centers.
    r1 : float
        Radius of the first sphere.
    r2 : float
        Radius of the second sphere.
    Returns
    -------
    vol: float
        Volume of the overlap between the two spheres.
    Notes
    -----
    See for example http://mathworld.wolfram.com/Sphere-SphereIntersection.html
    for more details.
    """
    vol = (math.pi / (12 * d) * (r1 + r2 - d)**2 *
           (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2))
    return vol / (4./3 * math.pi * min(r1, r2) ** 3)



def _blob_overlap(blob1, blob2):
    """Finds the overlapping area fraction between two blobs.
    Returns a float representing fraction of overlapped area.
    Parameters
    ----------
    blob1 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    blob2 : sequence of arrays
        A sequence of ``(row, col, sigma)`` or ``(pln, row, col, sigma)``,
        where ``row, col`` (or ``(pln, row, col)``) are coordinates
        of blob and ``sigma`` is the standard deviation of the Gaussian kernel
        which detected the blob.
    Returns
    -------
    f : float
        Fraction of overlapped area (or volume in 3D).
    """
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[:-1] - blob2[:-1])**2))
    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    if n_dim == 2:
        return _compute_disk_overlap(d, r1, r2)

    else:  # http://mathworld.wolfram.com/Sphere-SphereIntersection.html
        return _compute_sphere_overlap(d, r1, r2)



def _prune_blobs(blobs_array, overlap):
    """Eliminated blobs with area overlap.
    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 (or 4) values,
        ``(row, col, sigma)`` or ``(pln, row, col, sigma)`` in 3D,
        where ``(row, col)`` (``(pln, row, col)``) are coordinates of the blob
        and ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
        This array must not have a dimension of size 0.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.
    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)
    tree = spatial.cKDTree(blobs_array[:, :-1])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])



def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=.2,overlap=.5, log_scale=False):
    r"""Finds blobs in the given grayscale image.
    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.
    Parameters
    ----------
    image : 2D or 3D ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    Returns
    -------
    A : (n, image.ndim + 1) ndarray
        A 2d array with each row representing 3 values for a 2D image,
        and 4 values for a 3D image: ``(r, c, sigma)`` or ``(p, r, c, sigma)``
        where ``(r, c)`` or ``(p, r, c)`` are coordinates of the blob and
        ``sigma`` is the standard deviation of the Gaussian kernel which
        detected the blob.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian
    Examples
    --------

    array([[ 266.        ,  115.        ,   11.88888889],
           [ 263.        ,  302.        ,   17.33333333],
           [ 263.        ,  244.        ,   17.33333333],
           [ 260.        ,  174.        ,   17.33333333],
           [ 198.        ,  155.        ,   11.88888889],
           [ 198.        ,  103.        ,   11.88888889],
           [ 197.        ,   44.        ,   11.88888889],
           [ 194.        ,  276.        ,   17.33333333],
           [ 194.        ,  213.        ,   17.33333333],
           [ 185.        ,  344.        ,   17.33333333],
           [ 128.        ,  154.        ,   11.88888889],
           [ 127.        ,  102.        ,   11.88888889],
           [ 126.        ,  208.        ,   11.88888889],
           [ 126.        ,   46.        ,   11.88888889],
           [ 124.        ,  336.        ,   11.88888889],
           [ 121.        ,  272.        ,   17.33333333],
           [ 113.        ,  323.        ,    1.        ]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}\sigma` for
    a 2-D image and :math:`\sqrt{3}\sigma` for a 3-D image.
    """
    image = img_as_float(image)

    if log_scale:
        start, stop = log(min_sigma, 10), log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    #print(sigma_list)
    # computing gaussian laplace
    # s**2 provides scale invariance
    gl_images = [-gaussian_laplace(image, s) * s ** 2 for s in sigma_list]

    #cv2.imshow("gl", gl_images)
    image_cube = np.stack(gl_images, axis=-1)
    #print(image_cube)
    #print(image_cube.shape)
    local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                  footprint=np.ones((3,) * (image.ndim + 1)),
                                  threshold_rel=0.0,
                                  exclude_border=False)

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    p_blobs = _prune_blobs(lm, overlap)


    return  p_blobs


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

#Classiffy a Cell
class Cell():
    #initialize a Cell with location, radius, filename, and State
    def __init__(self, id, x, y, r, filename=None, is_healthy=None):
        self.id = id
        self.x = x
        self.y = y
        self.r = r
        self.filename = filename
        self.is_healthy = is_healthy

    #given a file name calc the max illuminate pixel in the image
    def readCell(self,filename):
        img = cv2.imread(os.path.join(cell_dir, filename), 1)  # load the img as just BGR COLOR SPACE
        avg = np.mean(img[:, :, 1])
        max = np.max(img[:, :, 1])
        threshold = avg + (.5 * avg)

        #Determine if Cells are healthy or not by the expression of protein
        if (max < threshold):
            cv2.imwrite(os.path.join('healthyCells/img' + str(self.id) + '.tif'), img)
            self.is_healthy = True
            return 0
        else:
            # unhealthy case
            cv2.imwrite(os.path.join('unhealthyCells/img' + str(self.id) + '.tif'), img)
            self.is_healthy = False
            return 1

#Edge check if directory for image split and storage exists
def makeDirectory():
    # delete current directory and make a new one
    if (os.path.isdir(cell_dir) == True):
        shutil.rmtree(cell_dir)
    os.mkdir(cell_dir)

    if (os.path.isdir(healthyDir) == True):
        shutil.rmtree(healthyDir)
    os.mkdir(healthyDir)

    if (os.path.isdir(unhealthyDir) == True):
        shutil.rmtree(unhealthyDir)
    os.mkdir(unhealthyDir)

    if (os.path.isdir(edgecells) == True):
        shutil.rmtree(edgecells)
    os.mkdir(edgecells)

    if (os.path.isdir(log_masks) == True):
        shutil.rmtree(log_masks)
    os.mkdir(log_masks)

#given a list of cells apply the readCells function
def readAllCells(listofcells):
    maxList = []
    for i in range(len(listofcells)):
        curr_cell = listofcells[i]
        maxList.append(curr_cell.readCell(curr_cell.filename)) #how to determine this threshold
    return sum(maxList)

#The Main Function
def LoG(img):

    makeDirectory() #clear working directories each time

    #Get the Original Image
    image = cv2.imread(img,1)

    #get the Sobal Edges
    image_gray = rgb2gray(image)

    #Get the Laplacian of the Gaussian
    print("Starting Laplacian of Gaussian")
    t0 = time()
    blobs_log = blob_log(image_gray, max_sigma=20, min_sigma=11, num_sigma=5, threshold=.01, overlap=0.3) #these params where chosen through manual testing
    t1 = time() - t0
    print("done in %0.3fs." % t1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(3)


    #ID of Cells
    id = 0

    #prepare data for iteration
    blobs_list = [blobs_log]
    colors = ['red']
    sequence = zip(blobs_list, colors)

    for _,(blobs, color) in enumerate(sequence):
        for blob in blobs:
            matrix = np.full(image.shape,255) #init a full white image

            y_o, x_o, r = blob #init blob

            r = int(math.ceil(r)) #radius of the determined blob
            y = int(y_o) #Column of determined blob
            x = int(x_o) #row of determined blob

            mask = sector_mask(matrix.shape, (y, x), r, (0, 360))
            matrix[~mask] = 0 #have the mask be black

            # show the output image
            # cv2.imshow("Output"+str(y), matrix)
            cv2.imwrite(os.path.join('log_masks/img' + str(id) + '.tif'), matrix)
            id+=1
    return t1,blobs_log

def Main():

    tests = os.listdir("./Train_Data")
    mAP = []
    mPVOC = []
    time_list = []
    min_dist_array = []

    #data arrays for circle based measurement
    min_dist_array_c = []
    mAP_c = []
    mPVOC_c = []

    for test in tests:
        print(test)
        image = os.listdir('./Train_Data/'+test+'/images')[0]
        print(image)
        # print(image)
        #print(os.path.join(dirname,test+'/images/'+image))
        #run LoG to get a folder of masks
        time,blob_list = LoG(os.path.join("./Train_Data",test+'/images/'+image))

        time_list.append(time)
        #doing mAP
        masks = os.listdir('./Train_Data/'+test+'/masks')
        pred_masks = os.listdir('./log_masks')
        #print(pred_masks)
        true_iou = []

        for mask in masks:
            if mask != "Mask_Total.png" and mask != "Circle_Masks": #we dont care about the total mask for LoG
                # calc the centroid of the mask
                print(mask)
                opened_mask = cv2.imread('./Train_Data/'+test+'/masks/'+ mask)
                #print(opened_mask.shape)
                mask_x, mask_y = centeroidnp(opened_mask)

                #for each centroid in blobs_list, calc euclid distance to maskx,masky
                #get the shortest euclid distance
                dist_list = []
                #print(blob_list)

                for row in blob_list:
                    dist = eucldist([row[0],row[1]],[mask_x,mask_y],opened_mask.shape[0],opened_mask.shape[1])
                    dist_list.append(dist)
                    min_dist_array.append(np.min(dist_list))
                all_iou = []
                for pred in pred_masks:
                    #print(mask)
                    iou = IOU(cv2.imread('./Train_Data/'+test+'/masks/'+mask), cv2.imread("./log_masks/"+pred))

                    #print(iou)
                    all_iou.append(iou)
                true_iou.append(max(all_iou))

        ap, PVOC = AP(np.asarray(true_iou))

        mAP.append(ap)
        mPVOC.append(PVOC)
        if (os.path.isdir('./Train_Data/' + test + '/masks/Circle_Masks') == True):

            #do the same now with circle masks
            c_masks = os.listdir('./Train_Data/' + test + '/masks/Circle_Masks')

            for mask in c_masks:

                # calc the centroid of the mask
                opened_mask = cv2.imread('./Train_Data/'+test+'/masks/Circle_Masks/'+ mask)
                print(mask)
                mask_x, mask_y = centeroidnp(opened_mask)

                #for each centroid in blobs_list, calc euclid distance to maskx,masky
                #get the shortest euclid distance
                dist_list = []
                #print(blob_list)

                for row in blob_list:
                    dist = eucldist([row[0],row[1]],[mask_x,mask_y],opened_mask.shape[0],opened_mask.shape[1])
                    dist_list.append(dist)
                    min_dist_array_c.append(min(dist_list))
                all_iou = []
                for pred in pred_masks:
                    #print(mask)
                    iou = IOT(cv2.imread('./Train_Data/'+test+'/masks/Circle_Masks/'+mask), cv2.imread("./log_masks/"+pred))

                    #print(iou)
                    all_iou.append(iou)
                true_iou.append(max(all_iou))

        ap, PVOC = AP(np.asarray(true_iou))

        mAP_c.append(ap)
        mPVOC_c.append(PVOC)

    with open('LOG_DATA.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        mdist = np.mean(min_dist_array)
        mdist_c = np.mean(min_dist_array_c)
        rows = zip(mAP,mPVOC,time_list)
        for row in rows:
            wr.writerow(row)
        wr.writerow(["mAP", "mPVOC", "Avg. Time", "Total Time","Avg. Normalized Distance Between Centroids", "mAP of Min. Circle packing", "PVOC of Min. Circle Packing","Avg. Normalized Distance Between Centroids (Min. Circle Packing)"])
        rows = [np.mean(mAP),np.mean(mPVOC),np.mean(time_list),np.sum(time_list),mdist, np.mean(mAP_c),np.mean(mPVOC_c),mdist_c]
        wr.writerow(rows)


def Detect_Cells(dir):
    print("Using LoG as a cell detection method ... ")


if __name__ == '__main__':

   Main()