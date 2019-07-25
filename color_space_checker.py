# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:47:33 2019
Last Modified on TODO

@author: Matias Ijäs
"""

import matplotlib.pyplot as plt
import os
from skimage import io

#filename = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facedataset/1_22_166_-165_51.JPG'
filename = 'C://Users/Matias Ijäs/Documents/Matias/data/exif-gps-samples/exif-gps-samples/DSCN0010.JPG'

# %% Functions

def mask_it(img, mmin, mmax):
    
    n_img = img.copy()
    xm, ym, zm = n_img.shape
    for x in range(0,xm):
        for y in range(0,ym):
            for z in range(0,zm):
                if n_img[x,y,z] < mmin or n_img[x,y,z] > mmax:
                    n_img[x,y,z] = 0

    return n_img

# %% Reading a file
"""    
img = plt.imread(filename)

# %% First method to find high intensity

# Just the color without other colors as zero
img_R = img[:,:,0]
img_G = img[:,:,1]
img_B = img[:,:,2]

# Plotting
plt.figure()
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray', vmin=150, vmax=255)
plt.subplot(2,2,2)
plt.imshow(img_R, cmap='gray', vmin=150, vmax=255)
plt.subplot(2,2,3)
plt.imshow(img_G, cmap='gray', vmin=150, vmax=255)
plt.subplot(2,2,4)
plt.imshow(img_B, cmap='gray', vmin=150, vmax=255)
plt.show()

# Second method of finding intensity over certain level

pic_R = img.copy()
pic_G = img.copy()
pic_B = img.copy()

pic_R[:,:,1] = 0
pic_R[:,:,2] = 0
pic_G[:,:,0] = 0
pic_G[:,:,2] = 0
pic_B[:,:,0] = 0
pic_B[:,:,1] = 0

# Creating masks and showing only high enough intensity and above

m_img = mask_it(img, 192, 256)
m_R = mask_it(pic_R, 192, 256)
m_G = mask_it(pic_G, 192, 256)
m_B = mask_it(pic_B, 192, 256)

m_img = mask_it(img, 150, 256)
m_R = mask_it(pic_R, 150, 256)
m_G = mask_it(pic_G, 150, 256)
m_B = mask_it(pic_B, 150, 256)

# Plotting
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(pic_R)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(pic_G)
plt.subplot(1,2,2)
plt.imshow(pic_B)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(m_img)
plt.subplot(1,2,2)
plt.imshow(m_R)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(m_G)
plt.subplot(1,2,2)
plt.imshow(m_B)
plt.show()
"""

# %% Reading from file and showing masked image

def read_and_mask_image(filename):
    img = plt.imread(filename)
    m_img = mask_it(img, 150, 256)
    
    """
    plt.figure()
    plt.imshow(m_img)
    """
    return m_img
    
def read_from_folder(read_folder, save_folder):
    
    for subdir, dirs, files in os.walk(read_folder):
        for file in files:
            
            if (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                print(file)
                filename = read_folder + '/' + file
                m_img = read_and_mask_image(filename)
                io.imsave('{}/{}'.format(save_folder, file), m_img)
                
save_folder = "C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_train"
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

read_from_folder("C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facedataset", save_folder)
# read_from_folder("C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/facetestset", save_folder)