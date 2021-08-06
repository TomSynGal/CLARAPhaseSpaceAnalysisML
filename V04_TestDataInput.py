# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 19:11:26 2021

@author: thoma
"""

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

OLDopticsParams = np.loadtxt('MATLABTrainingDataY.txt',delimiter=',') #For MATLAB Data
opticsParams = np.loadtxt('Ydata.txt',delimiter=',') #For Python Data
opticsParamsSwap = np.loadtxt('YdataSwap.txt',delimiter=',') #For Python Data

OLDsinograms = np.loadtxt('MATLABTrainingDataX.txt',delimiter=','); #For MATLAB Data
sinograms = np.loadtxt('Xdata.txt',delimiter=','); #For Python Data


'''
nimage = opticsParams.shape[0];
sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,24,-1,1));
print(sinograms.shape)
print(opticsParams.shape)
'''

