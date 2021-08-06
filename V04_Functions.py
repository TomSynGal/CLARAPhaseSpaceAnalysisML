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
    
    inTheBracket = betaX*epsilonX
    phaseRange = np.sqrt(inTheBracket)
    phaseRange = 3*phaseRange

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
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       
        
        #This didn't help the situation at all.
        #slicedImageProjection = np.exp(slicedImageProjection)
        
        sinogram[i] = slicedImageProjection
        
    return sinogram

##############################################################################
#
#
#
##############################################################################

def ArrayHeatMapPlot(inputData, savetag):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(savetag+".png")

##############################################################################
#
#
#
##############################################################################

def MakeTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples):
    
    Xdata = np.zeros((projections*samples, phaseResolution))

    Ydata =([alphaX, betaX, epsilonX])*(0.2+1.6*(np.random.rand(samples,3)))
    
    for i in range (samples):
    
        A = Ydata[i,0]
        B = Ydata[i,1]
        E = Ydata[i,2]

        singleSinogram = MakeSinogram(A, B, E, phaseResolution, projections)
    
        Xdata[i*projections:(i+1)*projections, :] = singleSinogram
       
    return Xdata, Ydata
    

##############################################################################
#
#
#
##############################################################################
    
