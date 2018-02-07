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
from numpy.linalg import norm

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'] 
act2 = ['Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan', 'Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']
men1 = ['Alec Baldwin', 'Bill Hader', 'Steve Carell','Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']
women1=['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Kristin Chenoweth', 'Fran Drescher', 'America Ferrera']

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def getData(actors,file):

    ''' Download all the images of a set of actors taken from a given file and stores the 
    bounding boxes for the face crops in a text file
    
    Arguments:
    actors --- an array containing the names of the actors whose images will be downloaded
    file --- a String filename of the textfile containing the URLs for the images of all the actors
    '''
    testfile = urllib.URLopener()
    
    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open(file):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                
                #take the sixth "word" in the line which correspond to the coordinates
                #and store it in a text file with the name of the image file so that it
                #can be referenced later
                coordinates = line.split()[5]
                f = open ('bounding_boxes.txt', 'a')
                f.write (name+str(i) + " " + coordinates + "\n")
                f.close () 
                
                print filename
                i += 1
  


## Run this to download all images
def downloadImages ():
    '''downloadImages downloads all images from both actor sets'''
    getData(act,"facescrub_actors.txt") #first set of actors
    getData(act,"facescrub_actresses.txt") #first set of actresses
    getData(act2,"facescrub_actors.txt") #second set of actors
    getData (act2,"facescrub_actresses.txt")#second set of actresses
    
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

def processData(actors):
    ''' Process all the downloaded images of a set of actors by cropping it 
    according to the proper bounding boxes, scaling it to a 32x32 images, and 
    turning it into its grayscale version
    
    Arguments:
    actors --- The array of actors whose images will be processed
    '''
    #store all the coordinates from the textfile in a dictionary; coordinates 
    #will be referenced later by filename
    dict = {}
    f = open ('bounding_boxes.txt', 'r')
    for line in f:
        name = line.split()[0]
        coordinates = line.split ()[1]
        dict [name] = coordinates
    f.close ()
    
    #read, crop, scale, grayscale, split the images
    for a in actors:
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
           
##Run this to process all downloaded images
def do_processData():
    '''Process images for both sets of actors'''
    processData (act) #crop, etc first set of actors and save in appropriate folders
    processData(act2) #crop, etc second set of actors and save in appropriate folders

def readImages (folder,N):
    '''Read the processed images and store the flattened 32x32 pixel images into an 1024xN array where 
    N is the number of images. Create the 1xN array containing the correct label for each of the images
    which are selected to be either 0 or 1. Return both arrays.
    
    Arguments: 
    folder --- the folder which the images should be read from (ie "Training" or "Validation")
    x --- the number of images per label which will be read
    
    Return:
    arr --- 1024xN array containing the flattened representation of images
    y --- 1xN array of labels of the images
    '''
    carell = []
    baldwin = []
    #read baldwin images
    for i in range (0,N):
        try:
            image = imread("sorted/"+ folder + "/baldwin"+ str(i) + ".png")
        except IOError as e:
            continue
        image = image[:,:,0] #take only 1 channel
        baldwin.append (image.flatten())
    #read carell images
    for i in range (0,N):
        try:
            image = imread("sorted/"+ folder + "/carell"+ str(i) + ".png")
        except IOError as e:
            continue
        image = image[:,:,0] #take only 1 channel
        carell.append (image.flatten())
    #arrange in array
    y = array ([0]*len(carell) + [1]*len(baldwin)) #array containing labels
    arr = np.concatenate((carell,baldwin),axis=0) #concatenate both arrays 

    arr = (arr + 0.)/255 #cast to float and scale to numbers from 0 to 1

    return (array(arr).T,array(y).T)

def f(x, y, theta):
    '''Taken from CSC411 course website'''
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    '''Taken from CSC411 course website'''
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha,max_iter):
    '''Modified from CSC411 course website'''
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def test(theta, folder):
    '''Tests the trained theta against a set of images. Returns the performance of the 
    classifier against the chosen set of images. 
    
    Arguments:
    theta --- The theta trained used gradient descent
    folder --- The set of images which should be used (ie "Training" or "Validation")
    
    Return:
    performance --- Ratio of correctly classified images to total images
    '''
    correct = 0 #counter storing the number of images classfied correctly 
    (images,y) = readImages (folder,100)
    images = vstack( (ones((1, images.shape[1])), images))
    
    for i in range (0,len(y)):
        if (y[i] == 0):
            print ("The image is of Steve Carell. Classified as: ")
        else:
            print ("The image is of Alec Baldwin. Classified as: ")
        
        classification = dot(theta, images.T[i])
        
        #0.5 is used as the boundary for the labels. If the correct label is 0 and the image 
        #was classified as <0.5, then it was classfied correctly. If the correct label is 1
        #and the image was classified as >0.5, then it was classified correctly.  
        if (classification < 0.5):
            if (y[i] == 0):
                correct = correct + 1
            print ("Carell " + str(classification) + "\n")
        else: 
            if (y[i] == 1):
                correct = correct + 1
            print ("Baldwin " + str(classification) + "\n")
    
    performance = float(float(correct)/len(y))*100
    print ("The classifier is correct " + str(performance) + "% of the time.")
    return performance
    
##Run this to execute Part 3 gradient descent and classfier performance
def Part3 ():
    (arr,y) = readImages("Training",74)
    print arr.shape
    print y.shape
    theta0 = array([float(0.001)]*1025) #theta is initialized to all 0.01
    alpha = 0.000001
    max_iter = 50000
    theta = grad_descent (f,df,arr,y,theta0,alpha,max_iter) #Run gradient descent to find theta
    
    performanceTraining = test (theta, "Training")#performance of theta on the training set
    performanceValidation = test (theta, "Validation")#performance of theta on the validation set

##Run this to execute Part4a of the assignment 
def part4a ():
    '''Display theta as a 32x32 pixel image when trained on 2 images vs a full training set'''
    (arr1, y1) = readImages ("Training", 2) #set with 2 images of each actor
    (arr2, y2) = readImages ("Training", 74) #set with all images
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    alpha = 0.000001
    max_iter = 30000
    theta = grad_descent (f,df,arr1,y1,theta0,alpha,max_iter) #grad on 2 images
    thetaFull = grad_descent (f,df,arr2,y2,theta0,alpha,max_iter)#grad on all imgs
    
    img1 = (theta[1:].reshape(32,32))
    img2 = (thetaFull[1:].reshape(32,32))
    
    plt.figure(0)
    plt.imshow(img1)
    plt.figure(1)
    plt.imshow(img2)
    
    plt.show (0) #show theta on 2 images
    plt.show (1) #show theta on all images

##Run this to execute Part 4b of the assignment
def part4b ():
    '''Represent theta as an image with various parameters. The number of iterations 
    run on gradient descent and initial theta values
    '''
    (arr,y) = readImages("Training",74) #read the full training set
    
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    theta01 = np.random.rand(1025) #initialized to random values in between 0 and 1
    alpha = 0.000001
    theta = np.empty([8,1025])
    theta[0] = grad_descent (f,df,arr,y,theta0,alpha,100) #100 iterations
    theta[1] = grad_descent (f,df,arr,y,theta0,alpha,500) #500 iterations
    theta[2]= grad_descent (f,df,arr,y,theta0,alpha,1000) #1000 iterations
    theta[3]= grad_descent (f,df,arr,y,theta0,alpha,5000) #5000 iterations
    theta[4] = grad_descent (f,df,arr,y,theta0, alpha,100)#0.01, 100 iter
    theta[5]= grad_descent (f,df,arr,y,theta01,alpha,100)#rand, 100 iter
    theta[6]= grad_descent (f,df,arr,y,theta0,alpha,1000)#0.01, 1000 iter
    theta[7]= grad_descent (f,df,arr,y,theta01,alpha,1000)#rand, 1000 iter
    
    theta = np.asarray(theta)
    
    img = np.empty([8,32,32])
    
    for i in range (0,8):
        img[i] = theta[i][1:].reshape(32,32)
        plt.figure (i)
        plt.imshow(img[i])
        plt.show(i)

def readImagesPt5 (folder, N,actors):
    '''Read the processed images and store the flattened 32x32 pixel images into an 1024xN array where 
    N is the number of images. Create the 1xN array containing the correct label for each of the images
    which are selected to be either 0 or 1. Return both arrays.
    
    Arguments: 
    folder --- the folder which the images should be read from (ie "Training" or "Validation")
    N --- the number of images per label which will be read
    actors --- the set of actors to read
    
    Return:
    arr --- 1024xN array containing the flattened representation of images
    y --- 1xN array of labels of the images
    '''
    women = []
    men = []
    
    for a in actors: #iterate through all the actors 
        success = 0
        name = a.split()[1].lower()
        for i in range (0,100):
            try:
                image = imread("sorted/"+ folder + "/" + name + str(i) + ".png")
            except IOError as e:
                continue
            image = image[:,:,0]
            success = success + 1
            if (success == N+1):
                break
            if (a in men1): #check if a is in the men or womens array
                men.append (image.flatten())
            else:
                women.append (image.flatten())
            
    #arrange in array
    y = array ([0]*len(men) + [1]*len(women))
    arr = np.concatenate((men,women),axis=0)
    
    arr = (arr + 0.)/255

    return (array(arr).T,array(y).T)

def testPt5(t, set,actors,numImages):
    '''Similar to test() but with classification as man/woman
    '''
    correct = 0
    (images,y) = readImagesPt5 (set,numImages,actors)
    images = vstack( (ones((1, images.shape[1])), images))
    for i in range (0,len(y)):
        # if (y[i] == 0):
        #     print ("man. Classified as: ")
        # else:
        #     print ("woman. Classified as: ")
        
        classification = dot(t, images.T[i])
        if (classification < 0.5):
            if (y[i] == 0):
                correct = correct + 1
            #print ("man " + str(classification) + "\n")
        else: 
            if (y[i] == 1):
                correct = correct + 1
            #print ("woman " + str(classification) + "\n")
    
    performance = float(float(correct)/len(y))*100
    print ("The classifier is correct " + str(performance) + "% of the time.")
    return performance

def run_tests (): 
    '''Train theta on a set of actors "act" and test the performance of theta 
    in classifying male vs female on "act" and another set of actors "act2" in 
    both their training and validation sets. Write the results in a text file.
    '''
    theta0 = array([float(0.01)]*1025) #theta is initialized to all low values
    #initialize the arrays containing the performances
    validationAct = []
    validationAct2 = []
    trainingAct = []
    
    #iterate through training set sizes from 5 to 65 in increments of 5
    for i in range(5,65,5):
        print ("Training set size: " + str(i))
        (arr,y) = readImagesPt5 ("Training",i,act)
        theta = grad_descent (f,df,arr,y,theta0, 0.000001, 30000)
        performance = testPt5(theta, "Validation",act,10)
        performance1 = testPt5(theta, "Validation",act2,10)
        performance2 = testPt5(theta,"Training",act,i)
        validationAct.append(performance) 
        validationAct2.append(performance1)
        trainingAct.append(performance2)
    
    #Write and save all the performances in a text file
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
        file.write (str(trainingAct [i]) + " ")
    file.write ("\n")
    
    file.close ()
    
## Part 5: run run_tests() then this function to show a plot of the performances in 
def plot ():
    '''Read the text file containing the performances of various tests and plot 
    them all on the same plot. Run this function after already performing the tests 
    but running the run_tests() function.
    '''
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

def fPart6 (X, Y, theta):
    '''Vectorized cost function'''
    return sum( (Y - matmul(theta.T,X)) ** 2)

def dfPart6 (X, Y, theta):
    '''Vectorized cost function gradient'''
    return 2*matmul(X,(matmul(theta.T, X)-Y).T)
    
def df_approxPt6d (X,Y,p,q,theta, h):
    cost = fPart6(X,Y,theta)
    theta[p][q] += h
    
    return (fPart6(X,Y,theta)-cost)/h


def readImages7 (folder):
    '''Read the processed images and store the flattened 32x32 pixel images. 
    Create array for labels and return two arrays
    
    Return:
    arr --- 1024xN array containing the flattened representation of images
    y --- 1xN array of labels of the images
    '''
    arr = []
    y = []
    for a in act: #iterate through all the actors 
        success = 0
        name = a.split()[1].lower()
        for i in range (0,100):
            try:
                image = imread("sorted/"+ folder + "/"+ name + str(i) + ".png")
            except IOError as e:
                continue
            image = image[:,:,0]
            success = success + 1
            if (success == 70):
                break
            arr.append(image.flatten())
            if (name == 'bracco'):
                y.append ([1,0,0,0,0,0])
            elif (name == 'gilpin'):
                y.append ([0,1,0,0,0,0])
            elif (name == 'harmon'):
                y.append ([0,0,1,0,0,0])
            elif (name == 'baldwin'):
                y.append ([0,0,0,1,0,0])
            elif (name == 'hader'):
                y.append ([0,0,0,0,1,0])
            elif (name == 'carell'):
                y.append ([0,0,0,0,0,1])
    
    arr = (array(arr)+ 0.)/255

    return (arr.T,array(y).T)    

##Run this to execute Part6d
def part6d (p,q):
    alpha = 0.000001
    max_iter = 100000
    theta0 = array([0.01]*1025*6).reshape([1025,6]) #theta is initialized to all low values

    (X,Y) = readImages7("Training")
    X = vstack( (ones((1, X.shape[1])), X))
    
    theta = dfPart6(X,Y,theta0)
    theta_approx = df_approxPt6d(X,Y,p,q,theta0,0.01)
    diff = (theta[p][q]-theta_approx)/theta[p][q]
    print("Approx: " + str(theta_approx))
    print("Exact: "+ str(theta[p][q]))
    print ("% Difference = " + str(diff*100))
    
'''
[1,0,0,0,0,0] = 'bracco
[0,1,0,0,0,0] = 'gilpin’
[0,0,1,0,0,0] = 'harmon'
[0,0,0,1,0,0] = ’baldwin’
[0,0,0,0,1,0] = ’hader’
[0,0,0,0,0,1] = ’carell’

'''  

##Run this to execute Part7
def Part7():
    alpha = 0.000001
    max_iter = 100000
    theta0 = array([0.01]*1025*6).reshape([1025,6]) #theta is initialized to all low values

    (X,Y) = readImages7("Training")
    X = vstack( (ones((1, X.shape[1])), X))
    
    theta = grad_descent(fPart6,dfPart6,X,Y,theta0,alpha,max_iter)
    
    resultTraining = dot(theta.T, X)
    correct = 0
    total = X.shape[1]
    for i in range (X.shape[1]):
        if (argmax(resultTraining[:,i]) == argmax(Y[:,i])):
            correct += 1

    print ("Accuracy for Training set = " +str((float(correct)/float(total))))
    
    (X,Y) = readImages7("Validation")
    X = vstack( (ones((1, X.shape[1])), X))
    resultValidation = dot (theta.T, X)
    correct = 0
    total = X.shape[1]
    for i in range (X.shape[1]):
        if (argmax(resultValidation[:,i]) == argmax(Y[:,i])):
            correct += 1

    print ("Accuracy for Validation set = " +str((float(correct)/float(total))))
  
##Run this to execute Part8
def Part8 ():
    alpha = 0.000001
    max_iter = 100000
    theta0 = array([0.01]*1025*6).reshape([1025,6]) #theta is initialized to all low values

    (X,Y) = readImages7("Training")
    X = vstack( (ones((1, X.shape[1])), X))
    
    img = np.empty([6,32,32])

    theta = grad_descent(fPart6,dfPart6,X,Y,theta0,alpha,max_iter)
    theta = theta.T
    for i in range (0,6):
        img[i] = theta[i][1:].reshape(32,32)
        plt.figure (i)
        plt.imshow(img[i])
        plt.show(i)



    
    
    
    