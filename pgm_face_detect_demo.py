# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:18:16 2019

@author: Matias Ijäs
"""

import cv2
from imageio import imread
import matplotlib.pyplot as plt

# %% Variables

file = 'C://Users/Matias Ijäs/Documents/Matias/data/ExtendedYaleB/yaleB16/yaleB16_P01A+000E+45.pgm'
casc_path = 'C://ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'

# %% Functions

def detect_face_area(filename, casc_path=casc_path):
    
    # This part uses mostly https://github.com/shantnu/FaceDetect
    img = imread(filename)
    faceCascade = cv2.CascadeClassifier(casc_path)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    number_faces = len(faces)
    print("Found {0} faces!".format(number_faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255), 2)
    
    """
    # Opens a face into another window
    cv2.imshow("Visible face", img)
    cv2.waitKey(0)
    """
    
    return img

# %% Script

img_orig = imread(file)
img_highlighted = detect_face_area(file, casc_path)

plt.imshow(img_orig, cmap='gray')
plt.figure()
plt.imshow(img_highlighted, cmap='gray')