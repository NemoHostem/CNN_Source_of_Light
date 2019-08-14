# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:09:52 2019
Last Modified on Fri May 24 14:30:00 2019

@author: Matias Ijäs
"""

import os
from exif import Image

# %% Changeable

folder = 'C://Users/Matias Ijäs/Documents/Matias/data/exif-gps-samples/exif-gps-samples'
# folder = 'C://Users/Matias Ijäs/Documents/Matias/data/coil-100'


# %% Do not change - Functions and main program

def read_exif_data(filename):

    with open(filename, 'rb') as img_file:
        img = Image(img_file)
    
    
    lat = img.gps_latitude
    lon = img.gps_longitude
    datet = img.datetime_original
    # info = dir(img)
    
    _lat = parse_lat_lon(lat)
    _lon = parse_lat_lon(lon)
    _date = parse_date(datet, 't')
    
    return _lat, _lon, _date

def parse_lat_lon(item):
    return item[0] + item[1]/100 + round(item[2],4)/10000

def parse_date(item, return_val):
    date, time = item.split()
    date_s = date.split(':')
    time_s = time.split(':')
    
    year, month, day = date_s
    hours, mins, secs = time_s
    
    text_format = day+'/'+month+'/'+year+' '+hours+':'+mins+':'+secs
    
    if return_val == 'y':
        return year
    elif return_val == 'm':
        return month
    elif return_val == 'd':
        return day
    elif return_val == 'h':
        return hours
    elif return_val == 'min':
        return mins
    elif return_val == 's':
        return secs
    elif return_val == 't':
        return text_format
    else:
        return -1
    
def loop_through_files(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.JPG') or file.endswith('.JPEG') or file.endswith('.PNG'):
                filename = folder + '/' + file
                print(read_exif_data(filename))


if os.path.exists(folder):
    print("Folder found! Looping through contents")
    loop_through_files(folder)
    
else:
    print("Folder does not exist.")