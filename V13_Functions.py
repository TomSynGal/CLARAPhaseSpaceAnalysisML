import numpy as np
from scipy import ndimage
from numpy.linalg import inv
import matplotlib.pyplot as plt
from V13_GlobalParameters import StaticPhaseRange
import h5py


def MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution):
    
    gammaX = (1 + np.square(alphaX)) / betaX

    covarMatx = np.array([[betaX, -alphaX],[-alphaX, gammaX]])
    
    covarMatx = epsilonX*covarMatx
    
    inverCovarMatx = inv(covarMatx)
    
    xvals = np.linspace(-1, 1, phaseResolution)*StaticPhaseRange
    
    pxvals = np.linspace(-1, 1, phaseResolution)*StaticPhaseRange
        
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


def MakeTomographyPhaseSpace(alphaX, betaX, epsilonX, phaseResolution):
    
    gammaX = (1 + np.square(alphaX)) / betaX

    covarMatx = np.array([[betaX, -alphaX],[-alphaX, gammaX]])
    
    covarMatx = epsilonX*covarMatx
    
    inverCovarMatx = inv(covarMatx)
    
    xvals = np.linspace(-1, 1, phaseResolution)*StaticPhaseRange
    
    pxvals = np.linspace(-1, 1, phaseResolution)*StaticPhaseRange
        
    phaseXV, phaseMomV = np.meshgrid(xvals, pxvals)
    
    x0 = StaticPhaseRange*(np.random.rand()-0.5)
    
    px0 = StaticPhaseRange*(np.random.rand()-0.5)
    
    phaseXV = phaseXV - x0
    
    phaseMomV = phaseMomV - px0
    
    phaseXV = phaseXV.transpose()
    
    phaseMomV = phaseMomV.transpose()
    
    phaseXV = phaseXV.reshape(1, phaseXV.size)
    
    phaseMomV = phaseMomV.reshape(1, phaseMomV.size)
    
    phaseSpaceVector = np.vstack([phaseXV, phaseMomV])
    
    transPhaseSpaceVector = phaseSpaceVector.transpose()
    
    phaseSpaceX = np.matmul(inverCovarMatx, phaseSpaceVector)
    phaseSpaceX = np.matmul(transPhaseSpaceVector, phaseSpaceX)
    phaseSpaceX = np.exp(np.diagonal(-phaseSpaceX/2)) 

    
    epsilonX_2 = 2*np.random.rand()*epsilonX
    
    alphaX_2 = 4*(np.random.rand()-0.5)*alphaX
    
    betaX_2 = (0.5+1.5*np.random.rand())*betaX
    
    gammaX_2 = (1 + np.square(alphaX_2)) / betaX_2
    
    covarMatx_2 = np.array([[betaX_2, -alphaX_2],[-alphaX_2, gammaX_2]])
    
    covarMatx_2 = epsilonX_2*covarMatx_2
    
    inverCovarMatx_2 = inv(covarMatx_2)
    
    x0_2 = StaticPhaseRange*(np.random.rand()-0.5)
    
    px0_2 = StaticPhaseRange*(np.random.rand()-0.5)
    
    phaseXV_2 = phaseXV - x0_2
    
    phaseMomV_2 = phaseMomV - px0_2
    
    phaseSpaceVector_2 = np.vstack([phaseXV_2, phaseMomV_2])
    
    transPhaseSpaceVector_2 = phaseSpaceVector_2.transpose()
    
    phaseSpaceX_2 = np.matmul(inverCovarMatx_2, phaseSpaceVector_2)
    phaseSpaceX_2 = np.matmul(transPhaseSpaceVector_2, phaseSpaceX_2)
    phaseSpaceX_2 = np.exp(np.diagonal(-phaseSpaceX_2/2))
    
    relint = 0.3 + 0.4*np.random.rand()
    
    totalPhaseSpaceX = (relint*phaseSpaceX)+((1-relint)*phaseSpaceX_2)
    
    totalPhaseSpaceX = totalPhaseSpaceX.reshape(phaseResolution, phaseResolution)
    
    return totalPhaseSpaceX


def MakeSinogram(alphaX, betaX, epsilonX, phaseResolution, projections):
    
    phaseSpaceX = MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram


def MakeTomographySinogram(alphaX, betaX, epsilonX, phaseResolution, projections):
    
    phaseSpaceX = MakeTomographyPhaseSpace(alphaX, betaX, epsilonX, phaseResolution)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram, phaseSpaceX


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


def MakeTomographyTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples):
    
    Xdata = np.zeros((projections*samples, phaseResolution))
    
    Ydata = np.zeros((samples, np.square(phaseResolution)))

    trainingParameters = np.array([epsilonX, betaX, (1+alphaX)])
    
    trainingParameters = np.diag(trainingParameters)
    
    MatX = 0.2+1.6*(np.random.rand(3, samples))
    
    trainingParameters = np.matmul(trainingParameters, MatX)
    
    trainingParameters = trainingParameters.transpose()

    for i in range (samples):
    
        print('Run Number')
        print(i)
    
        E = trainingParameters[i,0]
        B = trainingParameters[i,1]
        A = trainingParameters[i,2]

        singleSinogram, phaseSpaceX = MakeTomographySinogram(A, B, E, phaseResolution, projections)
    
        Xdata[i*projections:(i+1)*projections, :] = singleSinogram
        
        phaseSpaceX = np.reshape(phaseSpaceX, (np.square(phaseResolution)), order ='C') #Order could be wrong, 'C' 'F' 'A'
        
        Ydata[i,:] = phaseSpaceX
        
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)
    
    Ydata = Ydata*1000
    
    Ydata = Ydata.round(decimals=0)

    np.savetxt('XdataTomography.txt', Xdata, delimiter=',')

    np.savetxt('YdataTomography.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata


def ArrayHeatMapPlot(inputData, savetag):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(savetag+'.png')


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