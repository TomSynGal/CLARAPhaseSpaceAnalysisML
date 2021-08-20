import numpy as np
from V16_GlobalParameters import GlobalParams, TransformMatrix

ComplexPhaseSpace = False

def MacroPhaseSpace(aX, bX, eX, phaseResolution, Nparticles):
    
    if ComplexPhaseSpace:
        
        relint = 0.3 + 0.4*np.random.rand()
        
        xbins = pxbins = np.linspace(-1, 1, (phaseResolution+1))*GlobalParams['PhaseRange_Rec']
            
        phaseNormCoords = np.sqrt(eX)*np.random.normal(size = (2, int(Nparticles*relint)))
        
        normInverse = np.array([[np.sqrt(bX), 0.0],[(-aX/np.sqrt(bX)), (1/np.sqrt(bX))]])
        
        phaseCoords = np.matmul(normInverse,phaseNormCoords)
        
        x0 = GlobalParams['PhaseRange_Rec']*(np.random.rand()-0.5)
            
        px0 = GlobalParams['PhaseRange_Rec']*(np.random.rand()-0.5)
            
        phaseCoords[0] = phaseCoords[0,:] - x0
            
        phaseCoords[1] = phaseCoords[1,:] - px0
        
        eX_2 = 2*np.random.rand()*eX
            
        aX_2 = 4*(np.random.rand()-0.5)*aX
            
        bX_2 = (0.5+1.5*np.random.rand())*bX
    
        phaseNormCoords_2 = np.sqrt(eX_2)*np.random.normal(size = (2, int(Nparticles*(1-relint))))
        
        normInverse_2 = np.array([[np.sqrt(bX_2), 0.0],[(-aX_2/np.sqrt(bX_2)), (1/np.sqrt(bX_2))]])
        
        phaseCoords_2 = np.matmul(normInverse_2,phaseNormCoords_2)
        
        x0_2 = GlobalParams['PhaseRange_Rec']*(np.random.rand()-0.5)
            
        px0_2 = GlobalParams['PhaseRange_Rec']*(np.random.rand()-0.5)
            
        phaseCoords_2[0] = phaseCoords_2[0,:] - x0_2
            
        phaseCoords_2[1] = phaseCoords_2[1,:] - px0_2
            
        phaseCoords = np.append(phaseCoords, phaseCoords_2, 1)
        
        PhaseSpaceX, xedges, yedges = np.histogram2d(phaseCoords[0,:], phaseCoords[1,:], bins = [xbins,pxbins])
    
    else:
        
        #gX = (1+np.square(aX))/bX
        
        xbins = pxbins = np.linspace(-1, 1, (phaseResolution+1))*GlobalParams['PhaseRange_Rec']
        
        phaseNormCoords = np.sqrt(eX)*np.random.normal(size = (2, Nparticles))
    
        normInverse = np.array([[np.sqrt(bX), 0.0],[(-aX/np.sqrt(bX)), (1/np.sqrt(bX))]])
    
        phaseCoords = np.matmul(normInverse,phaseNormCoords)
        
        PhaseSpaceX, xedges, yedges = np.histogram2d(phaseCoords[0,:], phaseCoords[1,:], bins = [xbins,pxbins])
    
    return PhaseSpaceX, phaseCoords


def RealSinogram (aX, bX, eX, phaseResolution, Nparticles):
    
    Projections = TransformMatrix.shape[0]/2
    Projections = int(Projections)
    
    phaseSpaceX, phaseCoords = MacroPhaseSpace(aX, bX, eX, 
                                               phaseResolution, Nparticles)
    
    sinogram = np.zeros((Projections, phaseResolution))
    
    #xbins = np.linspace(-1, 1, (phaseResolution+1))*GlobalParams['PhaseRange_Rec']
    xbins = np.linspace(-1, 1, (phaseResolution+1))*GlobalParams['PhaseRangeMax_Obs']
    
    for i in range (Projections):
        j = 2*i
        k = j+2
        
        MiniMatrix = TransformMatrix[j:k, :]
        transformedCoords = np.matmul(MiniMatrix, phaseCoords)
        transPhaseSpaceDensity, edges = np.histogram(transformedCoords[0,:], bins = xbins)
        sinogram[i] = transPhaseSpaceDensity
        
    return sinogram, phaseSpaceX


def MakeRealImages(aX, bX, eX, phaseResolution, samples, Nparticles):
    
    Xdata = np.zeros((GlobalParams['RealProjections']*samples, phaseResolution))

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

        singleSinogram, phaseSpaceX = RealSinogram(A, B, E, phaseResolution, Nparticles)
    
        Xdata[i*GlobalParams['RealProjections']:(i+1)*GlobalParams['RealProjections'], :] = singleSinogram
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    np.savetxt('XdataReal.txt', Xdata, delimiter=',')

    np.savetxt('YdataReal.txt', Ydata, delimiter=',')

    return Xdata, Ydata

def MakeRealTomographyImages(aX, bX, eX, phaseResolution, samples, Nparticles):  
    
    Xdata = np.zeros((GlobalParams['RealProjections']*samples, phaseResolution))
    
    Ydata = np.zeros((samples, np.square(phaseResolution)))

    trainingParameters = np.array([eX, bX, (1+aX)])
    
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

        singleSinogram, phaseSpaceX = RealSinogram(A, B, E, phaseResolution, Nparticles)
    
        Xdata[i*GlobalParams['RealProjections']:(i+1)*GlobalParams['RealProjections'], :] = singleSinogram
        
        phaseSpaceX = np.reshape(phaseSpaceX, (np.square(phaseResolution)), order ='C') #Order could be wrong, 'C' 'F' 'A'
        
        Ydata[i,:] = phaseSpaceX
        
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)
    
    Ydata = Ydata*1000
    
    Ydata = Ydata.round(decimals=0)

    np.savetxt('XdataTomographyReal.txt', Xdata, delimiter=',')

    np.savetxt('YdataTomographyReal.txt', Ydata, delimiter=',')

    return Xdata, Ydata