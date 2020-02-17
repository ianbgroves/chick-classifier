#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:07:00 2019

@author: David
"""

import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
	help="path to the required directory")
args = vars(ap.parse_args())

#assign directory path
path = args["directory"]

#get last element of path to rename files
pathsplit = path.split('/')
filename = pathsplit[-1]

#set count variable
count = 0

#loop over files in directory, looking for jpgs
for f in os.listdir(path):
    if f.endswith('.jpg'):
        
        #set source and destination for new files
        dst = filename + '_' + str(count).zfill(3) + '.jpg'
        src = path + '/' + f
        dst = path + '/' + dst
        
        os.rename(src,dst)
        
        count += 1 
        