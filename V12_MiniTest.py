import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from V12_GlobalParameters import GlobalParams

def MacroPhaseSpace( alphaX, betaX, epsilonX, phaseResolution, Nparticles):
    
    gammaX = (1+np.square(alphaX))/betaX

    phaseRange = 3*np.sqrt(1.5)
    
    xbins = np.linspace(-1, 1, (phaseResolution+1))*phaseRange
    
    pxbins = np.linspace(-1, 1, (phaseResolution+1))*phaseRange
    
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

def MakeMacroMATXSinogram(alphaX, betaX, epsilonX, phaseResolution,
                 projections, Nparticles):
    
    phaseSpaceX, phaseCoords = MacroPhaseSpace(alphaX, betaX, epsilonX,
                                  phaseResolution, Nparticles)
    
    sinogram = np.zeros((projections, phaseResolution))
    
    for i in range(projections):
     
        angle = (i-1)*180/projections
        
        rotationMatrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        
        transformedCoords = np.matmul(rotationMatrix, phaseCoords)
        
        transPhaseSpaceDensity, edges = np.histogram(transformedCoords[0,:], bins = phaseResolution)
        
        sinogram[i] = transPhaseSpaceDensity
        
    return sinogram

def ArrayHeatMapPlot(inputData, savetag):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig(savetag+'.png')


MacroPhaseSpaceX, phaseCoords = MacroPhaseSpace(GlobalParams['AlphaX'], 
                             GlobalParams['BetaX'], 
                             GlobalParams['EpsilonX'], 
                             GlobalParams['PhaseResolution'],
                             GlobalParams['Nparticles'])

Figure1 = ArrayHeatMapPlot(MacroPhaseSpaceX, 'TestMacroPhaseSpace')

sinogramPhase1 = MakeMacroSinogram(GlobalParams['AlphaX'], 
                        GlobalParams['BetaX'], 
                        GlobalParams['EpsilonX'], 
                        GlobalParams['PhaseResolution'],
                        GlobalParams['Projections'],
                        GlobalParams['Nparticles'])

Figure2 = ArrayHeatMapPlot(sinogramPhase1, 'TestSinogramPhase1')

sinogramPhase2 = MakeMacroMATXSinogram(GlobalParams['AlphaX'], 
                        GlobalParams['BetaX'], 
                        GlobalParams['EpsilonX'], 
                        GlobalParams['PhaseResolution'],
                        GlobalParams['Projections'],
                        GlobalParams['Nparticles'])

Figure3 = ArrayHeatMapPlot(sinogramPhase2, 'TestSinogramPhase2')