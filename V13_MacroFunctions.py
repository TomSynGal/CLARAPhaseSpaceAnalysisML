import numpy as np
from scipy import ndimage
from numpy.linalg import inv
import matplotlib.pyplot as plt
from V13_GlobalParameters import StaticPhaseRange, TransformMatrix
import h5py

##############################################################################


##############################################################################

def MacroPhaseSpace( alphaX, betaX, epsilonX, phaseResolution, Nparticles):
    
    gammaX = (1+np.square(alphaX))/betaX
    
    xbins = np.linspace(-1, 1, (phaseResolution+1))*StaticPhaseRange
    
    pxbins = np.linspace(-1, 1, (phaseResolution+1))*StaticPhaseRange
    
    phaseNormCoords = np.sqrt(epsilonX)*np.random.normal(size = (2, Nparticles))

    normInverse = np.array([[np.sqrt(betaX), 0.0],[(-alphaX/np.sqrt(betaX)), (1/np.sqrt(betaX))]])

    thetaValues = np.linspace(0,2*np.pi, num = 101) #could be 100 or 99

    emitEllipseNormCos = np.array([np.cos(thetaValues)])

    emitEllipseNormSin = np.array([np.sin(thetaValues)])

    emitEllipseNorm = np.vstack((emitEllipseNormCos, emitEllipseNormSin))

    emitEllipseNorm = emitEllipseNorm*np.sqrt(2*epsilonX)

    phaseCoords = np.matmul(normInverse,phaseNormCoords)

    emitEllipse = np.matmul(normInverse,emitEllipseNorm)
    
    phaseSpaceDensity, xedges, yedges = np.histogram2d(phaseCoords[0,:], phaseCoords[1,:], bins = [xbins,pxbins])
    
    phaseSpaceX = phaseSpaceDensity
    
    return phaseSpaceX, phaseCoords


def MakeMacroSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles):
    
    phaseSpaceX, na = MacroPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram


def MakeTrainingImagesPhase1(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples, Nparticles):
    
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

        singleSinogram = MakeMacroSinogram(A, B, E, phaseResolution,
                                           projections, Nparticles)
    
        Xdata[i*projections:(i+1)*projections, :] = singleSinogram
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    np.savetxt('XdataMacroPhase1.txt', Xdata, delimiter=',')

    np.savetxt('YdataMacroPhase1.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata


def MakeMacroMATXSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles):
    
    phaseSpaceX, phaseCoords = MacroPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
        
    xbins = np.linspace(-1, 1, (phaseResolution+1))*StaticPhaseRange
    
    for i in range(projections):
     
        angle = (i-1)*3.14/projections
        
        rotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        
        transformedCoords = np.matmul(rotationMatrix, phaseCoords)
                
        transPhaseSpaceDensity, edges = np.histogram(transformedCoords[0,:], bins = xbins)
        
        sinogram[i] = transPhaseSpaceDensity
        
    return sinogram


def MakeTrainingImagesPhase2(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples, Nparticles):
    
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

        singleSinogram = MakeMacroMATXSinogram(A, B, E, phaseResolution,
                                           projections, Nparticles)
    
        Xdata[i*projections:(i+1)*projections, :] = singleSinogram
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    np.savetxt('XdataMacroPhase2.txt', Xdata, delimiter=',')

    np.savetxt('YdataMacroPhase2.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata


def RealSinogram (aX, bX, eX, phaseResolution, Nparticles):
    
    Projections = TransformMatrix.shape[0]/2
    Projections = int(Projections)
    
    phaseSpaceX, phaseCoords = MacroPhaseSpace(aX, bX, eX, 
                                               phaseResolution, Nparticles)
    
    sinogram = np.zeros((Projections, phaseResolution))
    
    xbins = np.linspace(-1, 1, (phaseResolution+1))*StaticPhaseRange
    
    for i in range (Projections):
        j = 2*i
        k = j+2
        
        MiniMatrix = TransformMatrix[j:k, :]
        transformedCoords = np.matmul(MiniMatrix, phaseCoords)
        transPhaseSpaceDensity, edges = np.histogram(transformedCoords[0,:], bins = xbins)
        #print(transPhaseSpaceDensity)
        sinogram[i] = transPhaseSpaceDensity
        
    return sinogram


def MakeRealImages(aX, bX, eX, phaseResolution, samples, Nparticles):
    
    Projections = TransformMatrix.shape[0]/2
    Projections = int(Projections)
    
    Xdata = np.zeros((Projections*samples, phaseResolution))

    Ydata = np.array([eX, bX, aX])
    
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

        singleSinogram = RealSinogram(A, B, E, phaseResolution, Nparticles)
    
        Xdata[i*Projections:(i+1)*Projections, :] = singleSinogram
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    np.savetxt('XdataMacroPhase2.txt', Xdata, delimiter=',')

    np.savetxt('YdataMacroPhase2.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata