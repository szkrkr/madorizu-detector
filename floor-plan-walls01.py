#Import packages
#import os
import cv2
import numpy as np
#import tensorflow as tf
#import sys

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")

# Import utilites
# from utils import label_map_util
# from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
FILE_NAME = 'madori01'
IMAGE_NAME = './original/' + FILE_NAME + '.jpg'
#Remove Small Items
im_gray = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 127
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#find all your connected components 
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_bw, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150  

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255
cv2.imwrite('./results/' + FILE_NAME + '-base.jpg', img2)

# cv2.imshow('room detector', img2)
#MorphologicalTransform
kernel = np.ones((6, 6), np.uint8)
dilation = cv2.dilate(img2, kernel)
cv2.imwrite('./results/' + FILE_NAME + '-dilate.jpg', dilation)
#erosion is no meaning
#erosion = cv2.erode(img2, kernel, iterations=6)
#cv2.imwrite('./results/' + FILE_NAME + '-erode.jpg', erosion)

#cv2.imshow("img2", img2)
#cv2.imshow("Dilation", dilation)
#cv2.imshow("Erosion", erosion)

findContours(dilation)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

