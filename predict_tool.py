# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:43:22 2019

@author: Matias Ijäs
"""

# %% Imports

from keras.models import load_model

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


# %% User defined variables

# Changeable
test_file = 'C://Users/Matias Ijäs/Documents/Matias/data/YaleB_gray/yaleB11_P00A+005E+10.jpg'
load_network_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/network'
load_network_filename = 'keras_regr_model_gray8_bw.h5'

# Do not change
max_angle = 180
min_angle = -180
filtered_min = 150
filtered_max = 256

# %% Functions

def rgb2gray(rgb):
    
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def rgb2filtered(img, mmin, mmax):
    
    n_img = img.copy()
    xm, ym, zm = n_img.shape
    for x in range(0,xm):
        for y in range(0,ym):
            for z in range(0,zm):
                if n_img[x,y,z] < mmin or n_img[x,y,z] > mmax:
                    n_img[x,y,z] = 0

    return n_img

def resize_gray_img(i_img, w, h):
    
    img = cv2.resize(i_img, (w,h), interpolation = cv2.INTER_AREA)
    return img

# %% Load existing model

model = load_model(load_network_folder + '/' + load_network_filename)

print("Model loaded succesfully.")
# Statistics about model structure
model.summary()
print("Input shape:", model.input_shape)
_, w, h, d = model.input_shape


# %% Test and visualize a single image and estimated light direction

img = plt.imread(test_file)
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the image as background
if d == 1:
    ax.imshow(img, extent=[-w/2,w/2,-h/2,h/2], cmap='gray')
else:
    ax.imshow(img, extent=[-w/2,w/2,-h/2,h/2])

img_shape = np.shape(img)
if len(img_shape) == 2 and d == 1: 
    # Test image gray, network gray
    img2 = resize_gray_img(img, w, h)
elif len(img_shape) == 2 and d == 3:
    # Test image gray, network RGB - does not work
    print("Error: test image is grayscale, network is RGB")
if len(img_shape) == 3 and d == 1:
    # Test image RGB, network gray
    img2 = rgb2gray(img)
    img2 = resize_gray_img(img2, w, h)
elif len(img_shape) == 3 and d == 3:
    # Test image RGB, network RGB
    img2 = cv2.resize(img, (w, h))
    
img2 = img2 / 255.0
img2 = img2.reshape((1, w, h, d))

# Compute evaluated angle and get real angle        
evaluated_angle = model.predict(img2)
evaluated_angle = evaluated_angle * (max_angle - min_angle) + min_angle

# Prepare x and y vectors to draw
if evaluated_angle < 90 and evaluated_angle > -90:
    x_range_e = np.array(range(0, int(w/2)))
else:
    x_range_e = -1 * np.array(range(0, int(w/2)))

print("Estimated:", evaluated_angle)

# Visualizing real and estimated values on top of image
y_range_e = np.array(math.tan((evaluated_angle / 180) * math.pi) * (x_range_e-0))

for i, y in enumerate(y_range_e):
    if y > h / 2:
        y_range_e[i] = h/2
        break
    elif y < -h / 2:
        y_range_e[i] = -h/2
        break
y_range_e = y_range_e[0:i+1]
x_range_e = x_range_e[0:i+1]
    
# Draw Estimated and Real lines on top of image
ax.plot(x_range_e, y_range_e, '--', linewidth=3, color='blue', label='Estimated')

ax.legend()