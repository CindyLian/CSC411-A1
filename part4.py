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
        if iter % 5000 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t


#Part 4a of the assignment (2 images vs all images)
def part4a ():
    (arr1, y1) = readImages ("Training", 2) #set with 2 images of each actor
    (arr2, y2) = readImages ("Training", 74) #set with all images
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    alpha = 0.000001

    theta = grad_descent (f,df,arr1,y1,theta0,alpha,30000) #grad on 2 images
    thetaFull = grad_descent (f,df,arr2,y2,theta0,alpha,30000)#grad on all imgs
    
    img1 = (theta[1:].reshape(32,32))
    img2 = (thetaFull[1:].reshape(32,32))
    
    plt.figure(0)
    plt.imshow(img1)
    plt.figure(1)
    plt.imshow(img2)
    
    plt.show (0)
    plt.show (1)

#Part 4bb of the assignment (100 iterations to 5000 iterations); random vs 0 init
def part4b ():
    (arr,y) = readImages("Training",74)
    
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    alpha = 0.000001 
    theta1  = grad_descent (f,df,arr,y,theta0,alpha,100) #100 iterations
    theta2  = grad_descent (f,df,arr,y,theta0,alpha,500) #500 iterations
    theta3  = grad_descent (f,df,arr,y,theta0,alpha,1000) #1000 iterations
    theta4  = grad_descent (f,df,arr,y,theta0,alpha,5000) #5000 iterations
    
    img1 = theta1[1:].reshape(32,32)
    img2 = theta2[1:].reshape(32,32)
    img3 = theta3[1:].reshape(32,32)
    img4 = theta4[1:].reshape(32,32)
    
    theta00 = array([float(0.01)]*1025) #theta is initialized to all low values
    theta01 = np.random.rand(1025)
    
    theta5 = grad_descent (f,df,arr,y,theta00, alpha, 100)
    theta6 = grad_descent (f,df,arr,y,theta01,alpha,100)
    theta7 = grad_descent (f,df,arr,y,theta00,alpha,1000)
    theta8 = grad_descent (f,df,arr,y,theta01,alpha,1000)
    img5 = theta5[1:].reshape(32,32)
    img6 = theta6[1:].reshape(32,32)
    img7 = theta7[1:].reshape(32,32)
    img8 = theta8[1:].reshape(32,32)

    plt.figure(0)
    plt.imshow(img1)
    plt.figure(1)
    plt.imshow(img2)
    plt.figure(2)
    plt.imshow(img3)
    plt.figure(3)
    plt.imshow(img4)
    plt.figure(4)
    plt.imshow(img5)
    plt.figure(5)
    plt.imshow(img6)
    plt.figure(6)
    plt.imshow(img7)
    plt.figure(7)
    plt.imshow(img8)
    
    plt.show (0)
    plt.show (1)
    plt.show (2)
    plt.show (3)
    plt.show (4)
    plt.show (5)
    plt.show (6)
    plt.show (7)