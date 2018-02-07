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

#carell vs baldwin, Assignment part 3
    
#flatten images into vectors of pixel intensities
#let Steve Carell = 0 and Alec Baldwin = 1
def readImages (folder,x):
    carell = []
    baldwin = []
    #read baldwin images
    for i in range (0,x):
        try:
            image = imread("sorted/"+ folder + "/baldwin"+ str(i) + ".png")
        except IOError as e:
            continue
        image = image[:,:,0]
        baldwin.append (image.flatten())
    #read carell images
    for i in range (0,x):
        try:
            image = imread("sorted/"+ folder + "/carell"+ str(i) + ".png")
        except IOError as e:
            continue
        image = image[:,:,0]
        carell.append (image.flatten())
    #arrange in array
    y = array ([0]*len(carell) + [1]*len(baldwin))
    arr = np.concatenate((carell,baldwin),axis=0)
    
    arr = (arr + 0.)/255

    return (array(arr).T,array(y).T)

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha,max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def do_grad ():
    (arr,y) = readImages("Training",74)
    theta0 = array([float(0.001)]*1025) #theta is initialized to all ones
    alpha = 0.000001
    max_iter = 50000
    theta = grad_descent (f,df,arr,y,theta0,alpha,max_iter)
    return theta
    
#if h <0.5, carell. If h > 0.5, baldwin
def test(t, set):
    correct = 0
    (validateImages,validateY) = readImages (set,100)
    validateImages = vstack( (ones((1, validateImages.shape[1])), validateImages))
    for i in range (0,len(validateY)):
        if (validateY[i] == 0):
            print ("The image is of Steve Carell. Classified as: ")
        else:
            print ("The image is of Alec Baldwin. Classified as: ")
        
        classification = dot(t, validateImages.T[i])
        if (classification < 0.5):
            if (validateY[i] == 0):
                correct = correct + 1
            print ("Carell " + str(classification) + "\n")
        else: 
            if (validateY[i] == 1):
                correct = correct + 1
            print ("Baldwin " + str(classification) + "\n")
    
    performance = float(float(correct)/len(validateY))*100
    print ("The classifier is correct " + str(performance) + "% of the time.")
    