import cv2
import numpy as np
import os
import cvlib as cv # pip install tensorflow, opencv-python, cvlib


# using opencv haar cascade
def face_detect(image, filter):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Read the input image
    if filter is not None:
        filter_image = filter(image)
    else:
        filter_image = image
    
    # detect faces
    face = face_cascade.detectMultiScale(filter_image, 1.1, 4)
    # display the output
    # when face is ()... it means no face detected
    if face is (): 
        print("No face detected")
        return None
    else:
        x, y, w, h = face[0].ravel()
        face_image = image[y:y+h, x:x+w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        return face_image

# using cvlib
def cv_face_detect(image):
    faces, confidences = cv.detect_face(image)
    if len(faces) == 0:
        print("No face detected")
        return None
    else:
        face_image = image[faces[0][1]:faces[0][3], faces[0][0]:faces[0][2]]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        return face_image

def gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def sobel_filter(img):
    dx = cv2.Sobel(img, -1, 1, 0, ksize=5)
    dy = cv2.Sobel(img, -1, 0, 1, ksize=5)
    
    abs_dx = cv2.convertScaleAbs(dx)
    abs_y = cv2.convertScaleAbs(dy)
    sobel = cv2.addWeighted(abs_dx, 0.5, abs_y, 0.5, 0)
    return sobel

def canny(img, threshold1=50, threshold2=150):
    return cv2.Canny(img, threshold1, threshold2)

if __name__=='__main__':
    # load image
    image_path = os.getcwd() 
    print(image_path) # /opt/ml/project/DataProcessing
    image = cv2.imread(image_path + '/../input/data/train/images/000001_female_Asian_45/mask1.jpg')
    print(image.shape) 
    face_image = face_detect(image)
    # cv2.imshow('image', face_image)