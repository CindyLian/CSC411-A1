from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
    
#turn an rgb photo to grayscale
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def processData(act):
    #store all the coordinates from the textfile in a dictionary
    #all the coordinates will be referenced by the name of the image
    dict = {}
    f = open ('bounding_boxes.txt', 'r')
    for line in f:
        name = line.split()[0]
        coordinates = line.split ()[1]
        dict [name] = coordinates
    f.close ()
    
    ##read, crop, scale, grayscale, split the images
    for a in act:
        name = a.split()[1].lower()
        i = 0
        count = 0
        while (count < 85):
            filename = name + str(i) 
            i = i + 1
            #read the uncropped images
            try:
                uncropped = imread("uncropped/" + filename + ".jpg")
            except IOError as e:
                continue
            
            #split the coordinates 
            coord = dict [filename]
            x1 = coord.split(',')[0]
            y1 = coord.split(',')[1]
            x2 = coord.split(',')[2]
            y2 = coord.split(',')[3]
            location = 0
            
            #sort it in the corresponding folder
            if (count < 65):
                location = "Training/"
            elif (count <75):
                location = "Test/"
            else:
                location = "Validation/"
            
            #crop, scale, grayscale, and save into new folder
            try: 
                cropped = uncropped[int(y1):int(y2), int(x1):int(x2)]
                scaled = imresize (cropped,(32,32))
                grayscaled = rgb2gray(scaled)
                imsave ("sorted/"+location + filename + ".png", grayscaled, cmap = cm.gray)
            except IndexError as e:
                continue
            
            count = count + 1
            print (filename + " " + str(count))
        
            