# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:02:57 2021

@author: thoma
"""
##############################################################################
#01/XX Brief
##############################################################################
'''

The purpose of this code is to generate data for emittance and optics 
measurements on CLARA, the compact linear accelerator at Daresbury lab.

Prior to this code, data creation occoured in MATLAB. In this study I will
transfer the same mathematics over to Python explainig steps within the code
as I go.

To run this code correctly, packages such as Numpy,Pandas, MatPlotLib, 
Tensorflow and Keras should be installed.

Also there is an accompanying script that contains all of the functions to run
this script.

This is the main script, all tasks should be performed from here, however they
do rely on the functions to run properly.

'''
##############################################################################
#End 01/XX Brief
##############################################################################
#
#
#
##############################################################################
#02/XX Imports
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import keras
import h5py
from numpy.linalg import inv
from V04_Functions import MakePhaseSpace, MakeSinogram, ArrayHeatMapPlot
from V04_Functions import MakeTrainingImages

##############################################################################
#End 02/XX Imports
##############################################################################
#
#
#
##############################################################################
#03/XX True/False Switchboard
##############################################################################

##############################################################################
#End 03/XX True/False Switchboard
##############################################################################
#
#
#
##############################################################################
#04/XX Global Parameters
##############################################################################

alphaX = -0.2

betaX = 1.5

epsilonX = 1.0#E-06

#epsilonX2 = 1.0

phaseResolution = 48

projections = 24

samples = 8000

##############################################################################
#End 04/XX Global Parameters
##############################################################################


phaseSpace = MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution)

sinogram = MakeSinogram(alphaX, betaX, epsilonX, phaseResolution, projections)

phaseSpaceFigure = ArrayHeatMapPlot(phaseSpace, 'PhaseFigure')

SinogramFigure = ArrayHeatMapPlot(sinogram, 'SinogramFigure')


##############################################################################
'''
#Data generation from Functions, currently crashes Python?


xData, yData = MakeTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                                  projections, samples)
'''
##############################################################################


##############################################################################

Xdata = np.zeros((projections*samples, phaseResolution))


#Old method
##########################################################################
#Ydata =([epsilonX, betaX, alphaX])*(0.2+1.6*(np.random.rand(samples,3)))
##########################################################################

#New method more in-line with the MATLAB way.
##################################################
Ydata = np.array([epsilonX, betaX, alphaX])
Ydata = np.diag(Ydata)
MatX = 0.2+1.6*(np.random.rand(3, samples))
Ydata = np.matmul(Ydata, MatX)
Ydata = Ydata.transpose()
##################################################


for i in range (samples):
    
    print('Run Number')
    print(i)
    
    E = Ydata[i,0]
    B = Ydata[i,1]
    A = Ydata[i,2]

    singleSinogram = MakeSinogram(A, B, E, phaseResolution, projections)
    
    Xdata[i*projections:(i+1)*projections, :] = singleSinogram
    
Xdata = Xdata*1000
Xdata = Xdata.round(decimals=0)


#Conventional Save
############################################### 
'''   
np.savetxt('Xdata.txt', Xdata, delimiter=',')

np.savetxt('Ydata.txt', Ydata, delimiter=',')
'''
###############################################

#HDF5 Save
##############################################################################

#with h5py.File('C:\Users\thoma\Documents\Daresbury Placement August 2021\Phase 2\RunProgramV04', 'w') as hdf:
#    hdf.create_dataset('Xdata', data = Xdata)
#    hdf.create_dataset('Ydata', data = Ydata)

##############################################################################

#HDF5 Save
##############################################################################
h5f = h5py.File('data.hf', 'w')
h5f.create_dataset('Xdata', data = Xdata)
h5f.create_dataset('Ydata', data = Ydata)
##############################################################################


##############################################################################