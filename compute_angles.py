# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:14:11 2019
Last Modified on Mon Aug 19 14:46 2019

@author: Matias Ijäs

This file computes radius and angles theta and phi for light direction in 2D image
from face3D generated images. Theta is used for estimating light direction in 2D space
and phi evaluates angle from 3D space to 2D surface.

The passport photos use the first standard function
The "twisted heads" use the second standard function
"""

import os
import math


def rotateX3D(theta, x, y, z):
    
    theta = theta * (math.pi / 180)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    new_x = x
    new_y = y * cos_t - z * sin_t
    new_z = z * cos_t + y * sin_t
    
    return new_x, new_y, new_z


def rotateY3D(theta, x, y, z):
    
    theta = theta * (math.pi / 180)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    new_x = x * cos_t - z * sin_t
    new_y = y
    new_z = z * cos_t + x * sin_t
    
    return new_x, new_y, new_z


def rotateZ3D(theta, x, y, z):
    
    theta = theta * (math.pi / 180)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    new_x = x * cos_t - y * sin_t
    new_y = y * cos_t + x * sin_t
    new_z = z
    
    return new_x, new_y, new_z


def compute_straight_face_angle(x,y,z):
    
    radius = 0
    phi = 0
    theta = 0
    
    radius = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y,x) * (180 / math.pi)
    theta = math.acos(z/radius) * (180 / math.pi)
    
    return radius, phi, theta


def compute_complicated_face_angle(x, y, z, pitch, yaw, roll):
    
    radius2 = 0
    phi2 = 0
    theta2 = 0
    
    # 3D space with no object rotation
    """
    # This method produces incorrect result for complicated angles (if any face angle =/= 0)
    radius = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x) * (180 / math.pi)
    theta = math.acos(z / radius) * (180 / math.pi)
    
    # This method produces incorrect result for complicated angles (if at least two face angles =/= 0)
    x_new = x * math.cos(roll) - y * math.sin(roll)
    y_new = y * math.cos(roll) + x * math.sin(roll)
    
    x_new2 = x_new * math.cos(yaw) - z * math.sin(yaw)
    z_new = z * math.cos(yaw) + x_new * math.sin(yaw)
    
    y_new2 = y_new * math.cos(pitch) - z_new * math.sin(pitch)
    z_new2 = z_new * math.cos(pitch) + y_new * math.sin(pitch)
    
    x, y, z = x_new2, y_new2, z_new2
    """
    
    # 3D space with object rotation
    x, y, z = rotateX3D(pitch, x, y, z)
    x, y, z = rotateY3D(yaw, x, y, z)
    x, y, z = rotateZ3D(roll, x, y, z)
    
    radius2 = math.sqrt(x**2 + y**2 + z**2)
    phi2 = math.atan2(y, x) * (180 / math.pi)
    theta2 = math.acos(z / radius2) * (180 / math.pi)
    
    # print(radius, phi, theta)
    print(radius2, phi2, theta2)
    print(x, y, z)
    
    return radius2, phi2, theta2


def separate_attributes(name):
    
    mode = -1
    n = 0
    x,y,z,pitch,yaw,roll = 0,0,0,0,0,0
    
    _name = name.split('.')
    attr = _name[0].split("_")
    if attr[0] == "test":
        return mode, n, x, y, z, pitch, yaw, roll
    elif len(attr) == 5:
        mode, n, x, y, z = attr
    elif len(attr) == 8:
        mode, n, x, y, z, pitch, yaw, roll = attr

    return int(mode), int(n), int(x), int(y), int(z), int(pitch), int(yaw), int(roll)


def read_values_from_folder(folder, result_file):
    
    data_stor = []
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            print(file)
            if (not file.endswith('test', 0, 4)) and (file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.jpg') or file.endswith('.jpeg')):
                filename = folder + '/' + file
                mode, n, x, y, z, pitch, yaw, roll = separate_attributes(file)
                # print(mode, n, x, y, z, pitch, yaw, roll)

                if mode == -1:
                    continue
                elif mode == 1 or mode == 2:
                    radius, phi, theta = compute_straight_face_angle(x,y,z)
                elif mode == 3:
                    radius, phi, theta = compute_complicated_face_angle(x, y, z, pitch, yaw, roll)
                else:
                    continue
                # print(filename, radius, phi, theta)
                data = ','.join(str(e) for e in [filename, radius, phi, theta, '\n'])
                data_stor.append(data)
    
    rf = open(result_file, 'w+')
    for data in data_stor:
        rf.write(data)
    rf.close()

# %% Reading data from file and saving to csv file
                   
from_folder = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_train_gray'
csv_file = 'C://Users/Matias Ijäs/Documents/Matias/face3d/examples/results/face_train_gray.csv'

read_values_from_folder(from_folder, csv_file)
                