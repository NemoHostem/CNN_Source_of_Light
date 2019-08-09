# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:36:57 2019

@author: Matias Ijäs
"""

# %% Imports

import os
import numpy as np
import cv2
from imageio import imread
from skimage import io
import matplotlib.pyplot as plt

# %% User variables

# Changeable
test_folder = 'C://Users/Matias Ijäs/Documents/Matias/data/ExtendedYaleB/yaleB11'
save_folder = 'C://Users/Matias Ijäs/Documents/Matias/data/YaleB_gray'
casc_path = 'C://ProgramData/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'

w, h, d = 128, 128, 1

# %% Functions

def visualize_pgm(test_file):
    
    img = imread(test_file)
    plt.imshow(img)

    
def read_pgm_files_from_folder(test_folder):
    
    test_files = []
    for subdir, dirs, files in os.walk(test_folder):
        for file in files:
            
            if ((file.endswith('.pgm') or file.endswith('.PGM')) and not (file.endswith('Ambient.pgm') or file.endswith('Ambient.PGM'))):
                print(file)
                filename = test_folder + '/' + file
                
                test_files.append(filename)
    
    return test_files


def collect_all_pgms(test_folder, save_folder):
    
    print("Adjusting save settings")
    create_save_folder(save_folder)
    test_files = read_pgm_files_from_folder(test_folder)
    num_test_files = len(test_files)
    print("Number of pgm files:", num_test_files)
    
    for i, file in enumerate(test_files, start=1):
        # visualize_pgm(file)
        print("File", file)
        _ = detect_face_area(file, casc_path)
        """
        img = detect_face_area(file, casc_path)
        img = resize_face_img(img, w, h, d)
        img = equalize_img(img, 1)
        filename = file.split('.')[0] + ".jpg"
        save_img(filename, img, save_folder)
        """
        if i % 10 == 0:
            print("Creating files :", round(((i)*100 / num_test_files),2), "% completed")

    print("Creating files :", round(((i)*100 / num_test_files),2), "% completed")


def detect_face_area(filename, casc_path=casc_path):
    
    # This part uses mostly https://github.com/shantnu/FaceDetect
    img = imread(filename)
    face_img = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(casc_path)
    print(casc_path)
    print(os.path.exists(casc_path))
    
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

    """
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    """
    
    if number_faces == 0:
        cv2.imshow("Faces not found", img)
        cv2.waitKey(0)
    else:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        cv2.imshow("Cropped face", face_img)
        cv2.waitKey(0)
    
    return face_img


def resize_face_img(face_img, w, h, d):
    
    img = np.array(face_img)
    img.resize(w,h,d)
    
    return img


def equalize_img(filename, status=0):
    
    if status:
        img = filename
    else:
        img = cv2.imread(filename, 0)
    img_eq = cv2.equalizeHist(img)
    
    return img_eq
    
    
def rgb2gray(rgb):
    
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def save_img(img_name, img, save_folder):

    io.imsave('{}/{}'.format(save_folder, img_name), img)
    

def create_save_folder(save_folder):
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    

# %% Script

collect_all_pgms(test_folder, save_folder)