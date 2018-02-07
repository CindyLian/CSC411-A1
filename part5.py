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

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'] 
act2 = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
men1 = ['Alec Baldwin', 'Bill Hader', 'Steve Carell','Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']
women1 = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']

def readImagesPt5 (folder, numImages,actors):
    women = []
    men = []
    
    for a in actors: 
        success = 0
        name = a.split()[1].lower()
        for i in range (0,100):
            try:
                image = imread("sorted/"+ folder + "/" + name + str(i) + ".png")
            except IOError as e:
                continue
            image = image[:,:,0]
            success = success + 1
            if (success == numImages+1):
                break
            if (a in men1):
                men.append (image.flatten())
            else:
                women.append (image.flatten())
            
    #arrange in array
    y = array ([0]*len(men) + [1]*len(women))
    arr = np.concatenate((men,women),axis=0)
    
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
        #if iter % 5000 == 0:
            #print "Iter", iter
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t
    
#if h <0.5, men. If h > 0.5, women
def test(t, set,actors,numImages):   
    correct = 0
    (validateImages,validateY) = readImagesPt5 (set,numImages,actors)
    validateImages = vstack( (ones((1, validateImages.shape[1])), validateImages))
    for i in range (0,len(validateY)):
        # if (validateY[i] == 0):
        #     print ("man. Classified as: ")
        # else:
        #     print ("woman. Classified as: ")
        
        classification = dot(t, validateImages.T[i])
        if (classification < 0.5):
            if (validateY[i] == 0):
                correct = correct + 1
            #print ("man " + str(classification) + "\n")
        else: 
            if (validateY[i] == 1):
                correct = correct + 1
            #print ("woman " + str(classification) + "\n")
    
    performance = float(float(correct)/len(validateY))*100
    print ("The classifier is correct " + str(performance) + "% of the time.")
    return performance

def run_tests ():    
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    validationAct = []
    validationAct2 = []
    trainingAct = []
    for i in range(5,65,5):
        print i
        (arr,y) = readImagesPt5 ("Training",i,act)
        theta = grad_descent (f,df,arr,y,theta0, 0.000001, 50000)
        performance = test (theta, "Validation",act,10)
        performance1 = test (theta, "Validation",act2,10)
        performance2 = test(theta,"Training",act,65)
        validationAct.append(performance) 
        validationAct2.append(performance1)
        trainingAct.append(performance2)
    
    file = open ("Part5.txt",'a')
    file.write ("validationAct ")
    for i in range (0,len(validationAct)):
        file.write (str(validationAct [i]) + " ")
    file.write ("\n")
    file.write ("validationAct2 ")
    for i in range (0,len(validationAct2)):
        file.write (str(validationAct2 [i]) + " ")
    file.write ("\n")
    file.write ("trainingAct ")
    for i in range (0,len(trainingAct)):
        file.write (str(trainngAct [i]) + " ")
    file.write ("\n")
    
    file.close ()
        
def plot ():
    file = open ("Part5.txt",'r')
    x = np.arange(5, 65, 5)

    y = np.zeros (shape=(3,12))
    count = 0
    for line in file:
        name = line.split()[0]
        for i in range (0,12):
            y[count][i] = line.split()[i+1]
        count = count + 1
        
    file.close ()
        
    plt.plot (x,y[0],label = "valicationAct")
    plt.plot (x,y[1], label = "validationAct2")
    plt.plot (x,y[2], label = "trainingAct")
    plt.legend()
    plt.xlabel ("Number of images per actor")
    plt.ylabel ("Performance")
    plt.title ("Performance vs size of training set")
    plt.show()

    
            
    
    