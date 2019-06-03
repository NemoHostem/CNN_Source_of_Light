# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:14:11 2019
Last Modified on Mon Jun  3 16:00:11 2019

@author: Matias Ijäs
"""

import math

def compute_straight_face_angle(x,y,z):
    radius = 0
    phi = 0
    theta = 0
    
    radius = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y,x)
    theta = math.acos(z/radius)
    
    return radius, phi, theta


def compute_complicated_angle(x, y, z, pitch, yaw, roll):
    radius = 0
    phi = 0
    theta = 0
    
    # 3D space with no object rotation
    radius = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.acos(z / radius)
    
    x_new = x * math.cos(pitch) - y * math.sin(pitch)
    y_new = y * math.cos(pitch) + x * math.sin(pitch)
    
    x_new2 = x_new * math.cos(yaw) - z * math.sin(yaw)
    z_new = z * math.cos(yaw) + x_new * math.sin(yaw)
    
    y_new2 = y_new * math.cos(roll) - z_new * math.sin(roll)
    z_new2 = z_new * math.cos(roll) + y_new * math.sin(roll)
    
    # 3D space with object rotation
    radius2 = math.sqrt(x_new2**2 + y_new2**2 + z_new2**2)
    phi2 = math.atan2(y_new2, x_new2)
    theta2 = math.acos(z_new2 / radius2)
    
    print(radius, phi, theta)
    print(radius2, phi2, theta2)
    
    return radius2, phi2, theta2


def separate_attributes(name):
    