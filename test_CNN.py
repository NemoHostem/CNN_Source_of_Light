# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:36:50 2019

@author: Matias Ijäs
"""

# %% Imports

from keras.models import load_model

#import os
import math
import csv
import numpy as np
from imageio import imread
from skimage.transform import rescale
import matplotlib.pyplot as plt

# %% User defined variables

# Changeable
num_testing = 5000
w, h, d = 128, 128, 3

test_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_test_gray'
test_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_PV_test_gray.csv'
save_stats_folder = 'C://Users/Matias Ijäs/Documents/Matias/results'
angle_stats_file = 'angle_stats.txt'
load_network_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/network'
load_network_filename = 'keras_regr_model_gray8_bw.h5'

# Do not change
max_angle = 180
min_angle = -180

# %% Functions (definitions)

def evaluate_accuracies(acc_count, sum_angle, est_val, real_val):
    
    if real_val < -90 and est_val > 90:
        err_val = abs(real_val + 360 - est_val)
    elif real_val > 90 and est_val < -90:
        err_val = abs(est_val + 360 - real_val)
    else:
        err_val = abs(real_val - est_val)
    sum_angle += err_val
    
    if err_val < 1:
        acc_count[0] += 1
    elif err_val < 5:
        acc_count[1] += 1
    elif err_val < 10:
        acc_count[2] += 1
    elif err_val < 30:
        acc_count[3] += 1
    elif err_val < 90:
        acc_count[4] += 1
    else:
        acc_count[5] += 1
        print(est_val, real_val)
        
    return acc_count, sum_angle

def do_n_tests_test_file(num_tests, test_each=False):
    
    # Accuracy (in degrees) <=1, <=5, <=10, <=30, <=90, >90
    accuracies_count = [0, 0, 0, 0, 0, 0]
    sum_angle_error = 0
    real_angles = np.zeros(num_tests)
    img_np = np.zeros((num_tests, w, h, d))
    indices = np.zeros(num_tests)
    
    for i in range(num_tests):
        if test_each:
            index = i
        else:
            index = np.random.randint(0,num_testing)
            
        imagepath = test_data[index][0]
        real_angles[i] = (float(test_data[index][2]) - min_angle) / (max_angle - min_angle)
        
        img = imread(imagepath)
        img = rescale(img, 1.0 / float(256.0/w), anti_aliasing=False)
        img = img / 255.0
        img_np[i] = img.reshape((w, h, d))
        indices[i] = index
    
    evaluated = model.predict(img_np)
    last_sum_angle = 0
    
    for i in range(num_tests):
        est_val = evaluated[i] * (max_angle - min_angle) + min_angle
        real_val = (real_angles[i]) * (max_angle - min_angle) + min_angle
        # print('Evaluated:', est_val, ', Real:', real_val)
        last_sum_angle = float(sum_angle_error)
        accuracies_count, sum_angle_error = evaluate_accuracies(accuracies_count, sum_angle_error, est_val, real_val)
        if last_sum_angle + 90 < sum_angle_error:
            print("Over 90 at: ", i)
        
    print("Accuracy count, <1, <5, <10, <30, <90, >90")
    print(accuracies_count)
    print("Average angle error: ", sum_angle_error / num_tests)
    
    return accuracies_count, sum_angle_error / num_tests

def test_all():
    
    num_tests = len(test_data)
    accuracies_count, avg_angle_error = do_n_tests_test_file(num_tests, test_each = True)
    
    return accuracies_count, avg_angle_error

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# %% Load existing model

model = load_model(load_network_folder + '/' + load_network_filename)

print("Model loaded succesfully.")
# Statistics about model structure
model.summary()
print("Input shape:", model.input_shape)
_, w, h, d = model.input_shape

# %% Load test_data

with open(test_file, newline='') as csvfile:
    test_data = list(csv.reader(csvfile))
    

# %% Prediction / testing

# Make predictions
num_tests = 1000
print("\nTesting", num_tests, "samples.")
ev = do_n_tests_test_file(num_tests)    # Test n samples
print("\nTesting all images from test_data")
ev2 = test_all()                        # Test whole folder

# %% Save average angle error statistics

stats_fname = save_stats_folder + "/" + angle_stats_file
with open(stats_fname, "a+") as f_err:
    
    f_err.write("\n\n" + load_network_filename)
    f_err.write("\nSubset " + str(num_tests) + ", Avg angle error: " + str(ev[1]) + ", ")
    f_err.write("Diviation " + str(ev[0]))
    f_err.write("\nAll" + ", Avg angle error: " + str(ev2[1]) + ", ")
    f_err.write("Diviation " + str(ev2[0]))

# %% Visualize and save test results
        
x_range = np.arange(6)
x_width = 0.35

fig, ax = plt.subplots(tight_layout=True)
p1 = ax.bar(x_range - x_width/2, ev[0], x_width, label='Subset')
p2 = ax.bar(x_range + x_width/2, ev2[0], x_width, label='All')

ax.set_ylabel('N')
ax.set_title('Error division in ' + load_network_filename)
ax.set_xticks(x_range)
ax.set_xticklabels(["<1", "<5", "<10", "<30", "<90", ">90"])
ax.legend()

autolabel(p1)
autolabel(p2)

plt.show()

print("Saving figure")
savefig_name = save_stats_folder + "/" + load_network_filename.split('.')[0] + ".jpg"
fig.savefig(savefig_name)
print("Save successful: Image", savefig_name, "saved.")

# %% Test and visualize a single image and estimated light direction

print("\nVisualizing single sample from test_data")
test_data_index = np.random.randint(0,num_testing)
sz = 256
img = plt.imread(test_data[test_data_index][0])
fig = plt.figure()
ax = fig.add_subplot(111)
# Plot the image as background
ax.imshow(img, extent=[-sz/2,sz/2,-sz/2,sz/2], cmap='gray')

img2 = rescale(img, 1.0 / float(256.0/w), anti_aliasing=False, anti_aliasing_sigma=None)
img2 = img2 / 255.0
img2 = img2.reshape((1, w, h, d))

# Compute evaluated angle and get real angle        
evaluated_angle = model.predict(img2)
evaluated_angle = evaluated_angle * (max_angle - min_angle) + min_angle
real_angle = float(test_data[test_data_index][2])

# Prepare x and y vectors to draw
if evaluated_angle < 90 and evaluated_angle > -90:
    x_range_e = np.array(range(0, int(sz/2)))
else:
    x_range_e = -1 * np.array(range(0, int(sz/2)))
    
if real_angle < 90 and real_angle > -90:
    x_range_r = np.array(range(0, int(sz/2)))  
else:
    x_range_r = -1 * np.array(range(0, int(sz/2)))

print("Estimated:", evaluated_angle, "Real:", real_angle, "\nError:", abs(evaluated_angle - real_angle))

# Visualizing real and estimated values on top of image
y_range_e = np.array(math.tan((evaluated_angle / 180) * math.pi) * (x_range_e-0))
y_range_r = np.array(math.tan((real_angle / 180) * math.pi) * (x_range_r-0))

for i, y in enumerate(y_range_e):
    if y > sz / 2:
        y_range_e[i] = sz/2
        break
    elif y < -sz / 2:
        y_range_e[i] = -sz/2
        break
y_range_e = y_range_e[0:i+1]
x_range_e = x_range_e[0:i+1]

for i, y in enumerate(y_range_r):
    if y > sz / 2:
        y_range_r[i] = sz/2
        break
    elif y < -sz / 2:
        y_range_r[i] = -sz/2
        break
y_range_r = y_range_r[0:i+1]
x_range_r = x_range_r[0:i+1]
    
# Draw Estimated and Real lines on top of image
ax.plot(x_range_e, y_range_e, '--', linewidth=3, color='blue', label='Estimated')
ax.plot(x_range_r, y_range_r, '--', linewidth=3, color='green', label='Real')

ax.legend()