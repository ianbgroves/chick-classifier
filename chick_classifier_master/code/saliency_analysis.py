# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:09:39 2020

@author: Ian_b
"""
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import PIL.Image
import keras.backend as K
from matplotlib import pylab as plt
from keras.models import load_model
import cv2
from keras import models
from saliency import GradientSaliency



# Load and compile the model
model = models.load_model('/Users/Ian_b/Downloads/Crappy.h5')
model.compile(loss='mean_squared_error', optimizer='adam')


#from guided_backprop import GuidedBackprop
#guided_bprop = GuidedBackprop(model) # A very expensive operation, which hackingly creates 2 new temp models



def show_image(image, grayscale = True, ax=None, title='img'):
    if ax is None:
        plt.figure()
    plt.axis('off')
    
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
    else:
        image = image + 127.5
        image = image.astype('uint8')
        
        plt.imshow(image)
        plt.title(title)
    
def load_image(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)
    
    return im - 127.5

# Load an image and make the prediction
img_path = '/Users/Ian_b/Downloads/CNNClassifier-master/hh11 BF.png'
img = load_image(img_path)
img = cv2.resize(img,(200,200))
x = np.reshape(img,(-1, 200,200, 1))
show_image(img, grayscale=True)

#x = np.reshape(img,(-1, 200,200, 1))
#x = np.expand_dims(img, axis=0)

preds = model.predict(x)
label = np.argmax(preds)
print(preds)

from integrated_gradients import IntegratedGradients
inter_grad = IntegratedGradients(model)

mask = inter_grad.get_mask(x[0])
show_image(mask, ax=plt.subplot('121'), title='integrated grad')

mask = inter_grad.get_smoothed_mask(x[0])
show_image(mask, ax=plt.subplot('122'), title='smoothed integrated grad')