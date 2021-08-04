# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:09:46 2021

@author: thoma
"""
##############################################################################
import numpy as np
import scipy.stats
from scipy import ndimage, misc
import itertools
from math import sqrt, isinf
from numpy.linalg import inv
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
##############################################################################
#
#
#
##############################################################################

def MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution):
    
    gammaX = (1 + np.square(alphaX)) / betaX
    
    phaseRange = 3*np.sqrt(betaX*epsilonX)

    covarMatx = np.array([[betaX, -alphaX],[-alphaX, gammaX]])
    
    covarMatx = epsilonX*covarMatx
    
    inverCovarMatx = inv(covarMatx)
    
    xvals = np.linspace(-1, 1, phaseResolution)*phaseRange
    
    pxvals = np.linspace(-1, 1, phaseResolution)*phaseRange
    
    #Calculation for xvals and pxvals has changed from the MATLAB code. Spacings
    #are calculated directly from the phase resolution which yields the exact same
    #array as doing it the original way in MATLAB. (This step took a while!)
    
    phaseXV, phaseMomV = np.meshgrid(xvals, pxvals)
    
    #I found a mesh grid!
    
    phaseXV = phaseXV.transpose()
    phaseMomV = phaseMomV.transpose()
    
    phaseXV = phaseXV.reshape(1, phaseXV.size)
    phaseMomV = phaseMomV.reshape(1, phaseMomV.size)
    
    phaseSpaceVector = np.vstack([phaseXV, phaseMomV])
    transPhaseSpaceVector = phaseSpaceVector.transpose()
    
    phaseSpaceX = np.matmul(inverCovarMatx, phaseSpaceVector)
    phaseSpaceX = np.matmul(transPhaseSpaceVector, phaseSpaceX)
    phaseSpaceX = np.exp(np.diagonal(-phaseSpaceX/2)) 
    
    phaseSpaceX = phaseSpaceX.reshape(phaseResolution, phaseResolution)
    
    return phaseSpaceX

##############################################################################
#
#
#
##############################################################################

def MakeSinogram(alphaX, betaX, epsilonX, phaseResolution, projections):
    
    phaseSpaceX = MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution)
    
    sinogram = np.zeros(projections, phaseResolution)
    
    for i in range(projections):
        
        angle = (i-1)*180/projections
        
        phaseSpaceXRotate = ndimage.rotate(phaseSpaceX, angle)
        
        phaseSpaceXProjection = phaseSpaceXRotate.sum(axis=0)
        
        sinogram[i] = phaseSpaceXProjection
        
        return phaseSpaceXProjection

##############################################################################
#
#
#
##############################################################################
    