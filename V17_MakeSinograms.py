import numpy as np
import matplotlib.pyplot as plt
from V17_Parameters import TransferMatrix



def PhaseSpaceSimple(params):
    
    betaX  = params['betaX']
    alphaX = params['alphaX']
    emitX  = params['emitX']
    
    phaseSpaceCoordsNorm = np.sqrt(emitX)*np.random.normal(size = (2, params['Nparticles']))
    
    normInverse = np.array([ [np.sqrt(betaX), 0.0], [(-alphaX/np.sqrt(betaX)), (1/np.sqrt(betaX))] ])
    #print(normInverse)
    
    phaseSpaceCoords = np.matmul(normInverse, phaseSpaceCoordsNorm)
    
    centroid = np.diag( [params['X0'], params['PX0']] )
    
    centroid = np.matmul(centroid, np.ones((2, params['Nparticles'])))
    
    phaseSpaceCoords = phaseSpaceCoords + centroid
        
    return phaseSpaceCoords

    
    
def Sinogram(params, phaseSpaceCoords):
    
    scanSteps = int( TransferMatrix.shape[0]/2 )
    
    sinogram    = np.zeros( (scanSteps, params['PhaseSpaceResolution']) )
    
    xbins = np.linspace(-1, 1, (params['PhaseSpaceResolution']+1))*params['PhaseSpaceRangeObs']*params['sigmaXObsMax']
    
    for i in range (scanSteps):
        
        j = 2*i
        k = j+2
        
        tm = TransferMatrix[j:k, :]
        
        transformedCoords = np.matmul(tm, phaseSpaceCoords)
        
        transPhaseSpaceDensity, edges = np.histogram(transformedCoords[0,:], bins = xbins)
        
        sinogram[i] = transPhaseSpaceDensity
        
    return sinogram



def MakeSinograms(params):
    
    if params['ComplexPhaseSpace']:
        
        Xdata, Ydata = MakeSinogramsComplexPS(params)
        
    else:
        
        Xdata, Ydata = MakeSinogramsSimplePS(params)
        
    return Xdata, Ydata
        
        

def MakeSinogramsSimplePS(params):
    
    samples = params['Samples']
    
    scanSteps = params['ScanSteps']
    
    Xdata = np.zeros((scanSteps*samples, params['PhaseSpaceResolution']))

    Ydata = np.array([params['emitX'], params['betaX'], params['alphaX']])
    
    Ydata = np.diag(Ydata)
    
    var   = 0.4 + 1.2*(np.random.rand(samples, 3))
    
    Ydata = np.matmul(var, Ydata)
    
    params1 = params.copy()
    
    for i in range (samples):
    
        print('', end=f'\rSample number: {i+1}/{samples}')
        
        params1['emitX']  = Ydata[i,0]
        params1['betaX']  = Ydata[i,1]
        params1['alphaX'] = Ydata[i,2]

        phaseSpaceCoords = PhaseSpaceSimple(params1)   
            
        Xdata[i*scanSteps:(i+1)*scanSteps, :] = Sinogram(params, phaseSpaceCoords)
        
    print(' Done!')
    
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)
    
    tag = params['Tag']

    np.savetxt('Sinograms' + tag + '.txt', Xdata, delimiter=',')

    np.savetxt('OpticsParameters' + tag + '.txt', Ydata, delimiter=',')

    return Xdata, Ydata



def MakeSinogramsComplexPS(params):
    
    samples = params['Samples']
    
    scanSteps = params['ScanSteps']
    
    psresolution = params['PhaseSpaceResolution']
    
    Xdata = np.zeros((scanSteps*samples, psresolution))
    
    sigmax  = np.sqrt(params['emitX']*params['betaX'])
    #print(sigmax)
    
    sigmapx = np.sqrt(params['emitX']*(1 + np.square(params['alphaX']))/params['betaX'])
    #print(sigmapx)
    
    xbins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmax

    pxbins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmapx
    
    Ydata = np.zeros( (psresolution*samples, psresolution) )

    npart  = params['Nparticles']
    
    params1 = params.copy()
        
    for i in range (samples):
    
        print('', end=f'\rSample number: {i+1}/{samples}')
        
        npart1 = int(npart*(0.3 + 0.2*(np.random.rand(1)[0]) ))
        npart2 = npart - npart1
        
        params1['X0']     = sigmax*(np.random.rand(1)[0]-0.5)*3
        params1['PX0']    = sigmapx*(np.random.rand(1)[0]-0.5)*3
        
        params1['emitX']  = params['emitX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaX']  = params['betaX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaX'] = params['alphaX']*(-1 + 3.0*(np.random.rand(1)[0]) )

        params1['Nparticles'] = npart1
        
        psCoords1 = PhaseSpaceSimple(params1)   

        params1['X0']     = sigmax*(np.random.rand(1)[0]-0.5)*3
        params1['PX0']    = sigmapx*(np.random.rand(1)[0]-0.5)*3
        
        params1['emitX']  = params['emitX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaX']  = params['betaX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaX'] = params['alphaX']*(-1 + 3.0*(np.random.rand(1)[0]) )

        params1['Nparticles'] = npart2
        
        psCoords2 = PhaseSpaceSimple(params1)

        phaseSpaceCoords = np.concatenate((psCoords1,psCoords2),axis=1)

        Xdata[i*scanSteps:(i+1)*scanSteps, :] = Sinogram(params, phaseSpaceCoords)
        
        PhaseSpaceDensity, xedges, yedges = np.histogram2d(phaseSpaceCoords[0,:], phaseSpaceCoords[1,:], bins = [xbins,pxbins])

        Ydata[i*psresolution:(i+1)*psresolution, :] = PhaseSpaceDensity
        
    print(' Done!')
        
    Xdata = Xdata*1000
    
    Xdata = Xdata.round(decimals=0)

    tag = params['Tag']

    np.savetxt('SinogramsComplexPS' + tag + '.txt', Xdata, delimiter=',')

    np.savetxt('PhaseSpaceDensity' + tag + '.txt', Ydata, delimiter=',')

    return Xdata, Ydata



def ArrayHeatMapPlot(inputData, fname):
    
    plt.clf()
    plt.imshow(inputData, cmap='hot', interpolation='nearest')
    plt.savefig(fname)
    plt.show()



def MultipleHeatMapPlot(image, figuresize, fname):
    
    plt.clf()
    fig = plt.figure(figsize=figuresize)
    
    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(image[i], interpolation='nearest', aspect='auto')
        plt.xticks([])
        plt.yticks([])
    
    plt.savefig(fname)
    plt.show()
    