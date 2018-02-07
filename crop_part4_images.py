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

x1 = 125
y1 = 50
x2 = 510
y2 = 440

img0a = imread ("Part 4/Figure0a.png")[y1:y2, x1:x2]
img1a = imread ("Part 4/Figure1a.png")[y1:y2, x1:x2]
img0b = imread ("Part 4/Figure0b.png")[y1:y2, x1:x2]
img1b = imread ("Part 4/Figure1b.png")[y1:y2, x1:x2]
img2b = imread ("Part 4/Figure2b.png")[y1:y2, x1:x2]
img3b = imread ("Part 4/Figure3b.png")[y1:y2, x1:x2]
img4b = imread ("Part 4/Figure4b.png")[y1:y2, x1:x2]
img5b = imread ("Part 4/Figure5b.png")[y1:y2, x1:x2]
img6b = imread ("Part 4/Figure6b.png")[y1:y2, x1:x2]
img7b = imread ("Part 4/Figure7b.png")[y1:y2, x1:x2]

imsave ("Part 4/cropped/Figure0a.png", img0a)
imsave ("Part 4/cropped/Figure1a.png", img1a)
imsave ("Part 4/cropped/Figure0b.png", img0b)
imsave ("Part 4/cropped/Figure1b.png", img1b)
imsave ("Part 4/cropped/Figure2b.png", img2b)
imsave ("Part 4/cropped/Figure3b.png", img3b)
imsave ("Part 4/cropped/Figure4b.png", img4b)
imsave ("Part 4/cropped/Figure5b.png", img5b)
imsave ("Part 4/cropped/Figure6b.png", img6b)
imsave ("Part 4/cropped/Figure7b.png", img7b)




