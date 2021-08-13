##############################################################################
# 01/10 Brief.
##############################################################################
'''
Insert Text Here.
'''
##############################################################################
# 01/10 End, Brief.
##############################################################################
#
#
#
##############################################################################
# 02/10 Imports.
##############################################################################
import numpy as np
from scipy import ndimage
from numpy.linalg import inv
import matplotlib.pyplot as plt
import h5py
##############################################################################
# 02/10 End, Imports.
##############################################################################
#
#
#
##############################################################################
# 03/10 Creation of the Phase Space Region.
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
# 03/10 End, Creation of the Phase Space Region.
##############################################################################
#
#
#
##############################################################################
# 04/10 Creation of the Tomography Phase Space Region.
##############################################################################

def MakeTomographyPhaseSpace(alphaX, betaX, epsilonX, phaseResolution):
    
    gammaX = (1 + np.square(alphaX)) / betaX

    phaseRange = 3*np.sqrt(1.5) #1.5 = nominal beam size^2 =  B*E

    covarMatx = np.array([[betaX, -alphaX],[-alphaX, gammaX]])
    
    covarMatx = epsilonX*covarMatx
    
    inverCovarMatx = inv(covarMatx)
    
    xvals = np.linspace(-1, 1, phaseResolution)*phaseRange
    
    pxvals = np.linspace(-1, 1, phaseResolution)*phaseRange
        
    phaseXV, phaseMomV = np.meshgrid(xvals, pxvals)
    
    x0 = phaseRange*(np.random.rand()-0.5)
    
    px0 = phaseRange*(np.random.rand()-0.5)
    
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
    
    x0_2 = phaseRange*(np.random.rand()-0.5)
    
    px0_2 = phaseRange*(np.random.rand()-0.5)
    
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

##############################################################################
# 04/10 End, Creation of the Tomography Phase Space Region.
##############################################################################
#
#
#
##############################################################################
# 05/10 Using Rotation and Density to Build a Sinogram.
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
# 05/10 End, Using Rotation and Density to Build a Sinogram.
##############################################################################
#
#
#
##############################################################################
# 06/10 Using Rotation and Density to Build a Tomography Sinogram.
##############################################################################

def MakeTomographySinogram(alphaX, betaX, epsilonX, phaseResolution, projections):
    
    phaseSpaceX = MakeTomographyPhaseSpace(alphaX, betaX, epsilonX, phaseResolution)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram, phaseSpaceX

##############################################################################
# 06/10 End, Using Rotation and Density to Build a Tomography Sinogram.
##############################################################################
#
#
#
##############################################################################
# 07/10 Generate Many Sinograms.
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
# 07/10 End, Generate Many Sinograms.
##############################################################################
#
#
#
##############################################################################
# 08/10 Generate Many Tomography Sinograms.
##############################################################################

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

##############################################################################
# 08/10 End, Generate Many Tomography Sinograms.
##############################################################################
#
#
#
##############################################################################
# 09/10 Generate a Plot.
##############################################################################

def ArrayHeatMapPlot(inputData, savetag):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(savetag+'.png')
    
##############################################################################
# 09/10 End, Generate a Plot.
##############################################################################
#
#
#
##############################################################################
# 10/10 Generate Multi-Plot.
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
# 10/10 End, Generate Multi-Plot.
##############################################################################
#
#
#
##############################################################################
# 10/10 Generate Multi-Plot.
##############################################################################

def MultipleHeatMapPlot2(AspectH, AspectV, image, totImage, VImage, HImage,
                        savetag):
    
    plt.clf()
    fig = plt.figure(figsize=(AspectH,AspectV))
    for i in range(totImage):
        sub = fig.add_subplot(VImage, HImage, i + 1)
        sub.imshow(image[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.savefig(savetag+'.png')
    
##############################################################################
# 10/10 End, Generate Multi-Plot.
##############################################################################
#
#
#
##############################################################################
# 11/11 Macroparticle Phase Space.
##############################################################################

def MacroPhaseSpace( alphaX, betaX, epsilonX, phaseResolution, Nparticles):
    
    gammaX = (1+np.square(alphaX))/betaX
    
    phaseNormCoords = np.sqrt(epsilonX)*np.random.normal(size = (2, Nparticles))

    normInverse = np.array([[np.sqrt(betaX), 0.0],[(-alphaX/np.sqrt(betaX)), (1/np.sqrt(betaX))]])

    thetaValues = np.linspace(0,2*np.pi, num = 101) #could be 100 or 99

    emitEllipseNormCos = np.array([np.cos(thetaValues)])

    emitEllipseNormSin = np.array([np.sin(thetaValues)])

    emitEllipseNorm = np.vstack((emitEllipseNormCos, emitEllipseNormSin))

    emitEllipseNorm = emitEllipseNorm*np.sqrt(2*epsilonX)

    phaseCoords = np.matmul(normInverse,phaseNormCoords)

    emitEllipse = np.matmul(normInverse,emitEllipseNorm)
    
    phaseSpaceDensity = np.histogram2d(phaseCoords[0,:], phaseCoords[1,:], bins = phaseResolution)

    phaseSpaceX = phaseSpaceDensity[0]
    
    return phaseSpaceX

##############################################################################
# 11/11 End, Macroparticle Phase Space.
##############################################################################
#
#
#
##############################################################################
# 12/12 Using Rotation and Density to Build a Sinogram With Macro Data.
##############################################################################

def MakeMacroSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles):
    
    phaseSpaceX = MacroPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotatedImage = ndimage.rotate(phaseSpaceX, angle, reshape=False, order=1)
        
        slicedImageProjection = rotatedImage.sum(axis=1)       

        sinogram[i] = slicedImageProjection
        
    return sinogram

##############################################################################
# 12/12 End, Using Rotation and Density to Build a Sinogram With Macro Data.
##############################################################################
#
#
#
##############################################################################
# 13/13 Generate Many Sinograms With Macro Data.
##############################################################################

def MakeMacroTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
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

##############################################################################
# 13/13 End, Generate Many Sinograms With Macro Data.
##############################################################################
#
#
#
##############################################################################
# 14/14 Macroparticle Phase Space With Matrix Transformation.
##############################################################################

def MacroMATXPhaseSpace( alphaX, betaX, epsilonX, phaseResolution, Nparticles):
    
    gammaX = (1+np.square(alphaX))/betaX
    
    phaseNormCoords = np.sqrt(epsilonX)*np.random.normal(size = (2, Nparticles))

    normInverse = np.array([[np.sqrt(betaX), 0.0],[(-alphaX/np.sqrt(betaX)), (1/np.sqrt(betaX))]])

    thetaValues = np.linspace(0,2*np.pi, num = 101) #could be 100 or 99

    emitEllipseNormCos = np.array([np.cos(thetaValues)])

    emitEllipseNormSin = np.array([np.sin(thetaValues)])

    emitEllipseNorm = np.vstack((emitEllipseNormCos, emitEllipseNormSin))

    emitEllipseNorm = emitEllipseNorm*np.sqrt(2*epsilonX)

    phaseCoords = np.matmul(normInverse,phaseNormCoords)

    emitEllipse = np.matmul(normInverse,emitEllipseNorm)
    
    phaseSpaceDensity = np.histogram2d(phaseCoords[0,:], phaseCoords[1,:], bins = phaseResolution)

    phaseSpaceX = phaseSpaceDensity[0]
    
    return phaseSpaceX, phaseCoords

##############################################################################
# 14/14 End, Macroparticle Phase Space With Matrix Transformation.
##############################################################################
#
#
#
##############################################################################
# 15/15 Using Matrix Transformation to Build a Sinogram With Macro Data.
##############################################################################

def MakeMacroMATXSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles):
    
    phaseSpaceX, phaseCoords = MacroMATXPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
    
  
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        
        transformedCoords = np.matmul(rotationMatrix, phaseCoords)
        
        transPhaseSpaceDensity = np.histogram2d(transformedCoords[0,:], transformedCoords[1,:], bins = phaseResolution)
        
        transPhaseSpaceDensity = transPhaseSpaceDensity[0]
        
        slicedTransformProjection = transPhaseSpaceDensity.sum(axis=1)       

        sinogram[i] = slicedTransformProjection
        
    return sinogram

##############################################################################
# 15/15 End, Using Matrix Transformation to Build a Sinogram With Macro Data.
##############################################################################
#
#
#
##############################################################################
# 16/16 Generate Many Sinograms With Matrix Transformation.
##############################################################################

def MakeMacroMATXTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
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

    np.savetxt('XdataMacroMATXPhase2.txt', Xdata, delimiter=',')

    np.savetxt('YdataMacroMATXPhase2.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata

##############################################################################
# 16/16 End, Generate Many Sinograms With Matrix Transformation.
##############################################################################
#
#
#
##############################################################################
# 17/17 Generate Many Sinograms With Real CLARA Transfer Matrices.
##############################################################################

def MakeREALMATRIXSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles, rotationMatrix):
    
    phaseSpaceX, phaseCoords = MacroMATXPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
    
  
    for i in range(projections):
     
        rotationMatrix = rotationMatrix
        
        transformedCoords = np.matmul(rotationMatrix, phaseCoords)
        
        transPhaseSpaceDensity = np.histogram2d(transformedCoords[0,:], transformedCoords[1,:], bins = phaseResolution)
        
        transPhaseSpaceDensity = transPhaseSpaceDensity[0]
        
        slicedTransformProjection = transPhaseSpaceDensity.sum(axis=1)       

        sinogram[i] = slicedTransformProjection
        
    return sinogram

##############################################################################
# 17/17 End, Generate Many Sinograms With Real CLARA Transfer Matrices.
##############################################################################
#
#
#
##############################################################################
# 18/18 Generate Many Sinograms With Real CLARA Data.
##############################################################################

def MakeREALMATRIXTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                       projections, samples, Nparticles, rotationMatrix):
    
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

        singleSinogram = MakeREALMATRIXSinogram(A, B, E, phaseResolution,
                                           projections, Nparticles,
                                           rotationMatrix)
    
        Xdata[i*projections:(i+1)*projections, :] = singleSinogram
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    np.savetxt('Xdata_REAL_Phase3.txt', Xdata, delimiter=',')

    np.savetxt('Ydata_REAL_Phase3.txt', Ydata, delimiter=',')
    
    '''
    h5f = h5py.File('data.hf', 'w')
    h5f.create_dataset('Xdata', data = Xdata)
    h5f.create_dataset('Ydata', data = Ydata)
    '''
    
    return Xdata, Ydata

##############################################################################
# 18/18 End, Generate Many Sinograms With Real CLARA Data.
##############################################################################