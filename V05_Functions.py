##############################################################################
# 01/XX Brief.
##############################################################################
'''
Insert Text Here.
'''
##############################################################################
# 01/XX End, Brief.
##############################################################################
#
#
#
##############################################################################
# 02/XX Imports.
##############################################################################
import numpy as np
from scipy import ndimage
from numpy.linalg import inv
import matplotlib.pyplot as plt
import h5py
##############################################################################
# 02/XX End, Imports.
##############################################################################
#
#
#
##############################################################################
# 03/XX Creation of the phase space region.
##############################################################################

def MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution):
    
    gammaX = (1 + np.square(alphaX)) / betaX

    phaseRange = 3*np.sqrt(1.5)

    covarMatx = np.array([[betaX, -alphaX],[-alphaX, gammaX]])
    
    covarMatx = epsilonX*covarMatx
    
    inverCovarMatx = inv(covarMatx)
    
    xvals = np.linspace(-1, 1, phaseResolution)*phaseRange
    
    pxvals = np.linspace(-1, 1, phaseResolution)*phaseRange
        
    phaseXV, phaseMomV = np.meshgrid(xvals, pxvals)
    
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
# 03/XX End, Creation of the phase space region.
##############################################################################
#
#
#
##############################################################################
# 04/XX Using rotation and density to build a sinogram.
##############################################################################

def MakeSinogram(alphaX, betaX, epsilonX, phaseResolution, projections):
    
    phaseSpaceX = MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram

##############################################################################
# 04/XX End, Using rotation and density to build a sinogram.
##############################################################################
#
#
#
##############################################################################
#
##############################################################################

def MakeTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples):
    
    Xdata = np.zeros((projections*samples, phaseResolution))

    Ydata = np.array([epsilonX, betaX, alphaX])
    
    Ydata = np.diag(Ydata)
    
    MatX = 0.2+1.6*(np.random.rand(3, samples))
    
    Ydata = np.matmul(Ydata, MatX)
    
    Ydata = Ydata.transpose()

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

    np.savetxt('Xdata.txt', Xdata, delimiter=',')

    np.savetxt('Ydata.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata

##############################################################################
#
##############################################################################
#
#
#
##############################################################################
#
##############################################################################

def ArrayHeatMapPlot(inputData, savetag):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(savetag+'.png')
    
##############################################################################
#
##############################################################################

def MultipleHeatMapPlot(AspectH, AspectV, image, savetag):
    
    plt.clf()
    fig = plt.figure(figsize=(AspectH,AspectV))
    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(image[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.savefig(savetag+'.png')
    
##############################################################################
#
##############################################################################