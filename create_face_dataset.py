"""
Created on Tue May 28 08:32:02 2019
Last Modified on May 31 10:03:05 2019

@author: Matias Ijäs

Parts of the original code from: https://github.com/YadiraF/face3d

This code is used to create a face dataset with single face 3d model and different lightning and rotation

You need to download YadiraF's face3d from github,
then add this file to face3d/examples

Depending on your libraries, you might need to use mesh_numpy in every part of face3d and change
original face3d imports such as line 27 in this file:
from face3d import mesh_numpy as mesh
"""

import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
import random
import math

sys.path.append('..')
from face3d import mesh_numpy as mesh

# %%

"""
light_test and transform_test are from https://github.com/YadiraF/face3d
"""

def light_test(vertices, light_positions, light_intensities, h = 256, w = 256):
	lit_colors = mesh.light.add_light(vertices, triangles, colors, light_positions, light_intensities)
	image_vertices = mesh.transform.to_image(vertices, h, w)
	rendering = mesh.render.render_colors(image_vertices, triangles, lit_colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering

def transform_test(vertices, obj, camera, h = 256, w = 256):
	'''
	Args:
		obj: dict contains obj transform paras
		camera: dict contains camera paras
	'''
	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
	
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)

	rendering = mesh.render.render_colors(image_vertices, triangles, colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering

def light_trans_test(vertices, obj, camera, light_positions, light_intensities, h = 256, w = 256):

	R = mesh.transform.angle2matrix(obj['angles'])
	transformed_vertices = mesh.transform.similarity_transform(vertices, obj['s'], R, obj['t'])
    
	if camera['proj_type'] == 'orthographic':
		projected_vertices = transformed_vertices
		image_vertices = mesh.transform.to_image(projected_vertices, h, w)
	else:

		## world space to camera space. (Look at camera.) 
		camera_vertices = mesh.transform.lookat_camera(transformed_vertices, camera['eye'], camera['at'], camera['up'])
		## camera space to image space. (Projection) if orth project, omit
		projected_vertices = mesh.transform.perspective_project(camera_vertices, camera['fovy'], near = camera['near'], far = camera['far'])
		## to image coords(position in image)
		image_vertices = mesh.transform.to_image(projected_vertices, h, w, True)
        
	lit_colors = mesh.light.add_light(vertices, triangles, colors, light_positions, light_intensities)
	rendering = mesh.render.render_colors(image_vertices, triangles, lit_colors, h, w)
	rendering = np.minimum((np.maximum(rendering, 0)), 1)
	return rendering

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

# %%--------- load mesh data
    
"""
load mesh data part is from https://github.com/YadiraF/face3d
"""

C = sio.loadmat('Data/example1.mat') # 3D face mesh
vertices = C['vertices']; 
global colors
global triangles
colors = C['colors']; triangles = C['triangles']
colors = colors/np.max(colors)
# move center to [0,0,0]
vertices = vertices - np.mean(vertices, 0)[np.newaxis, :]
s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
R = mesh.transform.angle2matrix([0, 0, 0]) 
t = [0, 0, 0]
vertices = mesh.transform.similarity_transform(vertices, s, R, t) # transformed vertices

# %% save settings

save_folder = 'results/facesettests'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
options = '-delay 12 -loop 0 -layers optimize' # gif. need ImageMagick.

# %%---- start lights

"""
Example of light positions and axis from https://github.com/YadiraF/face3d
"""

"""
# 1. fix light intensities. change light positions.
# x axis: light from left to right
light_intensities = np.array([[1, 1, 1]])
for i,p in enumerate(range(-200, 201, 40)): 
	light_positions = np.array([[p, 0, 300]])
	image = light_test(vertices, light_positions, light_intensities) 
	io.imsave('{}/test_1_{:0>2d}.jpg'.format(save_folder, i), image)
# y axis: light from up to down
for i,p in enumerate(range(200, -201, -40)): 
	light_positions = np.array([[0, p, 300]])
	image = light_test(vertices, light_positions, light_intensities) 
	io.imsave('{}/test_2_{:0>2d}.jpg'.format(save_folder, i), image)
# z axis: light near down to far
for i,p in enumerate(range(100, 461, 40)): 
	light_positions = np.array([[0, 0, p]])
	image = light_test(vertices, light_positions, light_intensities) 
	io.imsave('{}/test_3_{:0>2d}.jpg'.format(save_folder, i), image)
"""

# %%---- angles start

"""
Example of angles and scale from https://github.com/YadiraF/face3d
"""

"""
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'
# scale

for factor in np.arange(0.5, 1.2, 0.1):
	obj['s'] = scale_init*factor
	obj['angles'] = [0, 0, 0]
	obj['t'] = [0, 0, 0]
	image = transform_test(vertices, obj, camera) 
	io.imsave('{}/test_4_{:.2f}.jpg'.format(save_folder, factor), image)
"""


# %% Create a random dataset of rotated faces with different lightning

"""
obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'

n = 3000

for i in range(n):
    rand_x = random.randint(-250,250)
    rand_y = random.randint(-250,250)
    rand_z = random.randint(-50, 500)
    rand_R = 0.8 + random.random() * 0.2
    rand_G = 0.8 + random.random() * 0.2
    rand_B = 0.8 + random.random() * 0.2
    rand_pitch = random.randint(-90, 90)
    rand_yaw = random.randint(-90, 90)
    rand_roll = random.randint(-90, 90)
    
    obj['s'] = scale_init
    obj['angles'] = [rand_pitch, rand_yaw, rand_roll]
    obj['t'] = [0, 0, 0]
        
    light_intensities = np.array([[rand_R, rand_G, rand_B]])
    light_positions = np.array([[rand_x, rand_y, rand_z]])
    image = light_trans_test(vertices, obj, camera, light_positions, light_intensities) 
    io.imsave('{}/3_{:0>5d}_{}_{}_{}_{}_{}_{}.jpg'.format(save_folder, i, rand_x, rand_y, rand_z, rand_pitch, rand_yaw, rand_roll), image)
    
    radius = math.sqrt(rand_x**2 + rand_y**2 + rand_z**2)
    if radius == 0:
        continue
    phi = math.atan2(rand_y, rand_x)
    theta = math.acos(rand_z / radius)
    phi_deg = phi * (180 / math.pi)
    theta_deg = theta * (180 / math.pi)
    
    light_x = radius * math.sin(theta) * math.cos(phi)
    light_y = radius * math.sin(theta) * math.sin(phi)
    light_z = radius * math.cos(theta)
    
    print(rand_x, rand_y, rand_z, rand_pitch, rand_yaw, rand_roll)
    print("Phi, theta", phi_deg, theta_deg, light_x, light_y, light_z)
    
"""
        
# %% Create a random dataset of non-rotated faces with lightning from front

"""        
n = 1000

for i in range(n):
    rand_x = random.randint(-250,250)
    rand_y = random.randint(-250,250)
    rand_z = random.randint(50, 500)
    rand_R = 0.8 + random.random() * 0.2
    rand_G = 0.8 + random.random() * 0.2
    rand_B = 0.8 + random.random() * 0.2
    
    light_intensities = np.array([[rand_R, rand_G, rand_B]])
    light_positions = np.array([[rand_x, rand_y, rand_z]])
    image = light_test(vertices, light_positions, light_intensities) 
    io.imsave('{}/1_{:0>4d}_{}_{}_{}.jpg'.format(save_folder, i, rand_x, rand_y, rand_z), image)
    
    angle = math.atan2(rand_y,rand_x)
    deg = angle * (180 / math.pi)
    print(rand_x, rand_y, rand_z, deg)
    
"""

# %% Create a random set of non-rotated faces with lightning from behind
   
""" 
n = 1000

for i in range(n):
    rand_x = random.randint(-250,250)
    rand_y = random.randint(-250,250)
    rand_z = random.randint(-50, 50)
    rand_R = 0.8 + random.random() * 0.2
    rand_G = 0.8 + random.random() * 0.2
    rand_B = 0.8 + random.random() * 0.2
    
    light_intensities = np.array([[rand_R, rand_G, rand_B]])
    light_positions = np.array([[rand_x, rand_y, rand_z]])
    image = light_test(vertices, light_positions, light_intensities) 
    io.imsave('{}/2_{:0>4d}_{}_{}_{}.jpg'.format(save_folder, i, rand_x, rand_y, rand_z), image)
    
    angle = math.atan2(rand_y,rand_x)
    deg = angle * (180 / math.pi)
    print(rand_x, rand_y, rand_z, deg)
"""
    
# %% Test of rotation and computed angle

"""
test_i = 12

obj['s'] = scale_init
obj['angles'] = [-45, 0, 0]
obj['t'] = [0, 0, 0]
    
light_intensities = np.array([[1,1,1]])
light_positions = np.array([[0, 0, 300]])
image = light_trans_test(vertices, obj, camera, light_positions, light_intensities) 
io.imsave('{}/test_{}.jpg'.format(save_folder, test_i), image)

angle = math.atan2(rand_y,rand_x)
deg = angle * (180 / math.pi)
print(rand_x, rand_y, rand_z, deg)
"""


# %% Create test files to validate angles on complicated head twists

obj = {}
camera = {}
### face in reality: ~18cm height/width. set 180 = 18cm. image size: 256 x 256
scale_init = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1])) # scale face model to real size
## 1. fix camera model(stadard camera& orth proj). change obj position.
camera['proj_type'] = 'orthographic'

n = 10

for i in range(n):
    pitch = 45 - 5*i
    yaw = 0
    roll = 10*i
    x = 0.0
    y = 0.0
    z = 200.0
    
    obj['s'] = scale_init
    obj['angles'] = [pitch, yaw, roll]
    obj['t'] = [0, 0, 0]
        
    light_intensities = np.array([[1,1,1]])
    light_positions = np.array([[x,y,z]])
    image = light_trans_test(vertices, obj, camera, light_positions, light_intensities) 
    io.imsave('{}/test_5_{:0>2d}_{}_{}_{}.jpg'.format(save_folder, i, pitch, yaw, roll), image)
    
    """
    x_new = x * math.cos(roll) - y * math.sin(roll)
    y_new = y * math.cos(roll) + x * math.sin(roll)
    
    x_new2 = x_new * math.cos(yaw) - z * math.sin(yaw)
    z_new = z * math.cos(yaw) + x_new * math.sin(yaw)
    
    y_new2 = y_new * math.cos(pitch) - z_new * math.sin(pitch)
    z_new2 = z_new * math.cos(pitch) + y_new * math.sin(pitch)
    
    x, y, z = x_new2, y_new2, z_new2
    """
    
    x, y, z = rotateX3D(pitch, x, y, z)
    x, y, z = rotateY3D(yaw, x, y, z)
    x, y, z = rotateZ3D(roll, x, y, z)
    
    # 3D space with object rotation
    radius2 = math.sqrt(x**2 + y**2 + z**2)
    phi2 = math.atan2(y, x) * (180 / math.pi)
    theta2 = math.acos(z / radius2) * (180 / math.pi)
    
    print(pitch, yaw, roll)
    print("Phi, theta", phi2, theta2, "coords", x, y, z)

# %% 