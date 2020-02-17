# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:03:41 2020

@author: Ian_b
"""
import sys
import numpy as numpy
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import time
#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import regularizers
from keras import models
# SKLEARN
from sklearn.model_selection import train_test_split


# load the trained convolutional neural network


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications import VGG16

from vis.losses import ActivationMaximization
from vis.regularizers import TotalVariation, LPNorm
from vis.input_modifiers import Jitter
from vis.optimizer import Optimizer
from vis.callbacks import GifGenerator

# Build the VGG16 network with ImageNet weights
model = models.load_model('/Users/Ian_b/Downloads/Crappy.h5')
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
output_class = [20]

losses = [
    (ActivationMaximization(layer_dict[layer_name], output_class), 2),
    (LPNorm(model.input), 10),
    (TotalVariation(model.input), 10)
]
opt = Optimizer(model.input, losses)
opt.minimize(max_iter=500, verbose=True, input_modifiers=[Jitter()], callbacks=[GifGenerator('opt_progress')])

#TODO install Vis
#TODO build a model with a layer name