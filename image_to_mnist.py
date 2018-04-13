# create an array where we can store our 4 pictures
import numpy as np
import cv2
import math
from scipy import ndimage

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def convert(x):
    #images = np.zeros((1,784))
    # and the correct values
    #correct_vals = np.zeros((4,10))

    # we want to test our images which you saw at the top of this page
    #i = 0

    # read the image
    #gray = cv2.imread("seven.png", cv2.IMREAD_GRAYSCALE)
    gray = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    # resize the images and invert it (black background)
    gray = cv2.resize(255-gray, (28, 28))
    
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    # save the processed images
    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')


    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    #flatten = gray.flatten() / 255.0
    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for the first digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """
    #images[i] = flatten
    #correct_val = np.zeros((10))
    #correct_val[no] = 1
    #correct_vals[i] = correct_val
    #i += 1

    """
    the prediction will be an array with four values,
    which show the predicted number
    """
    """
    we want to run the prediction and the accuracy function
    using our generated arrays (images and correct_vals)
    """
    cv2.imwrite("mnist.png", gray)
    
    return gray

    #print sess.run(prediction, feed_dict={x: images, y_: correct_vals})
    #print sess.run(accuracy, feed_dict={x: images, y_: correct_vals})

