import cv2
import numpy
import sys
import os

import matplotlib.pyplot as plt

def face_detect(image):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    # display the output
    x, y, w, h = face.ravel()
    face_image = image[y:y+h, x:x+w]
    
    return face_image

if __name__=='__main__':
    # load image
    image_path = os.getcwd() 
    print(image_path) # /opt/ml/project/DataProcessing
    image = cv2.imread(image_path + '/../input/data/train/images/000001_female_Asian_45/mask1.jpg')
    print(image.shape) 
    face_image = face_detect(image)
    # cv2.imshow('image', face_image)