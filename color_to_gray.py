# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:32:49 2019

@author: Matias Ijäs
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from skimage import io


# %% Functions

def rgb2gray(rgb):
    
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def read_and_filter_image(filename):
    img = mpimg.imread(filename)
    g_img = rgb2gray(img)
    
    """
    plt.figure()
    plt.imshow(m_img)
    """
    return g_img
    
def read_from_folder(read_folder, save_folder):
    
    for subdir, dirs, files in os.walk(read_folder):
        for file in files:
            
            if (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                print(file)
                filename = read_folder + '/' + file
                m_img = read_and_filter_image(filename)
                io.imsave('{}/{}'.format(save_folder, file), m_img)
                
# %% Reading from file and showing masked image
                
save_folder = "C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_test_gray"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

read_from_folder("C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facetestset", save_folder)
# read_from_folder("C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facetestset", save_folder)