import numpy as np
import matplotlib.pyplot as plt
from V19_4D_Parameters import TransferMatrixX, TransferMatrixY



def PhaseSpaceSimple4D(params):
    
    betaX  = params['betaX']
    alphaX = params['alphaX']
    emitX  = params['emitX']
    
    betaY  = params['betaY']
    alphaY = params['alphaY']
    emitY  = params['emitY']
    
    emitScale = np.diag([emitX, emitX, emitY, emitY])
    
    psCoordsNorm = np.random.normal(size = (4, params['Nparticles']))
    
    phaseSpaceCoordsNorm = np.matmul(np.sqrt(emitScale), psCoordsNorm)
    
    normInverse = np.array([ [np.sqrt(betaX), 0.0, 0.0, 0.0],
                            [(-alphaX/np.sqrt(betaX)), (1/np.sqrt(betaX)), 0.0, 0.0], 
                            [0.0, 0.0, np.sqrt(betaY), 0.0],
                            [0.0, 0.0, (-alphaY/np.sqrt(betaY)), (1/np.sqrt(betaY))]])
    #print(normInverse)
    
    
    phaseSpaceCoords = np.matmul(normInverse, phaseSpaceCoordsNorm)
    
    centroid = np.diag([params['X0'], params['PX0'], params['Y0'], params['PY0']])
    
    centroid = np.matmul(centroid, np.ones((4, params['Nparticles'])))
    
    phaseSpaceCoords = phaseSpaceCoords + centroid
        
    return phaseSpaceCoords
  
  

def Sinogram4D(params, phaseSpaceCoords):
    
    coordsX = phaseSpaceCoords[0:2]
    
    coordsY = phaseSpaceCoords[2:4]
    
    scanSteps = int( TransferMatrixX.shape[0]/2 )
    
    sinogramX = np.zeros( (scanSteps, params['PhaseSpaceResolution']) )
    
    sinogramY = sinogramX
    
    xbins = np.linspace(-1, 1, (params['PhaseSpaceResolution']+1))*params['PhaseSpaceRangeObs']*params['sigmaXObsMax']

    ybins = np.linspace(-1, 1, (params['PhaseSpaceResolution']+1))*params['PhaseSpaceRangeObs']*params['sigmaYObsMax']
    
    for i in range (scanSteps):
        
        j = 2*i
        k = j+2
        
        tmX = TransferMatrixX[j:k, :]

        tmY = TransferMatrixY[j:k, :]
        
        transformedCoordsX = np.matmul(tmX, coordsX)
        
        transformedCoordsY = np.matmul(tmY, coordsY)
        
        transPhaseSpaceDensityX, edges = np.histogram(transformedCoordsX[0,:], bins = xbins)
        
        sinogramX[i] = transPhaseSpaceDensityX

        transPhaseSpaceDensityY, edges = np.histogram(transformedCoordsY[0,:], bins = ybins)
        
        sinogramY[i] = transPhaseSpaceDensityY
        
    return sinogramX, sinogramY



def MakeSinograms4D(params):
    
    if params['ComplexPhaseSpace']:
        
        Xdata, Xdata2, Ydata, Ydata2 = MakeSinogramsComplexPS4D(params)
        
    else:
        
        Xdata, Xdata2, Ydata, Ydata2 = MakeSinogramsSimplePS4D(params)
        
    return Xdata, Xdata2, Ydata, Ydata2



def MakeSinogramsSimplePS4D(params):
    
    samples = params['Samples']
    
    scanSteps = params['ScanSteps']
    
    Xdata = np.zeros((scanSteps*samples, params['PhaseSpaceResolution']))

    Xdata2 = np.array([params['emitX'], params['betaX'], params['alphaX']])
    
    Xdata2 = np.diag(Xdata2)

    Ydata = np.zeros((scanSteps*samples, params['PhaseSpaceResolution']))

    Ydata2 = np.array([params['emitY'], params['betaY'], params['alphaY']])
    
    Ydata2 = np.diag(Ydata2)
    
    var   = 0.4 + 1.2*(np.random.rand(samples, 3))
    
    Xdata2 = np.matmul(var, Xdata2)
    
    Ydata2 = np.matmul(var, Ydata2)
    
    params1 = params.copy()
    
    for i in range (samples):
    
        print('', end=f'\rSample number: {i+1}/{samples}')
        
        params1['emitX']  = Xdata2[i,0]
        params1['betaX']  = Xdata2[i,1]
        params1['alphaX'] = Xdata2[i,2]

        params1['emitY']  = Ydata2[i,0]
        params1['betaY']  = Ydata2[i,1]
        params1['alphaY'] = Ydata2[i,2]

        phaseSpaceCoords = PhaseSpaceSimple4D(params1)
            
        Xdata[i*scanSteps:(i+1)*scanSteps, :], Ydata[i*scanSteps:(i+1)*scanSteps, :] = Sinogram4D(params, phaseSpaceCoords)
        
    print(' Done!')
    
    Xdata = Xdata*1000

    Ydata = Ydata*1000
    
    Xdata = Xdata.round(decimals=0)

    Ydata = Ydata.round(decimals=0)
    
    tag = params['Tag']

    np.savetxt('SinogramsX' + tag + '.txt', Xdata, delimiter=',')

    np.savetxt('OpticsParametersX' + tag + '.txt', Xdata2, delimiter=',')

    np.savetxt('SinogramsY' + tag + '.txt', Ydata, delimiter=',')

    np.savetxt('OpticsParametersY' + tag + '.txt', Ydata2, delimiter=',')

    return Xdata, Xdata2, Ydata, Ydata2



def MakeSinogramsComplexPS4D(params):
    
    samples = params['Samples']
    
    scanSteps = params['ScanSteps']
    
    psresolution = params['PhaseSpaceResolution']
    
    Xdata = np.zeros((scanSteps*samples, psresolution))

    Ydata = np.zeros((scanSteps*samples, psresolution))
    
    sigmax  = np.sqrt(params['emitX']*params['betaX'])
    #print(sigmax)
    
    sigmay  = np.sqrt(params['emitY']*params['betaY'])
    #print(sigmay)
    
    sigmapx = np.sqrt(params['emitX']*(1 + np.square(params['alphaX']))/params['betaX'])
    #print(sigmapx)

    sigmapy = np.sqrt(params['emitY']*(1 + np.square(params['alphaY']))/params['betaY'])
    #print(sigmapy)
    
    xbins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmax

    pxbins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmapx

    ybins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmay

    pybins = np.linspace(-1, 1, (psresolution+1))*params['PhaseSpaceRange']*sigmapy
    
    Xdata2 = np.zeros( (psresolution*samples, psresolution) )

    Ydata2 = np.zeros( (psresolution*samples, psresolution) )

    npart  = params['Nparticles']
    
    params1 = params.copy()
        
    for i in range (samples):
    
        print('', end=f'\rSample number: {i+1}/{samples}')
        
        npart1 = int(npart*(0.3 + 0.2*(np.random.rand(1)[0]) ))
        npart2 = npart - npart1
        
        params1['X0']     = sigmax*(np.random.rand(1)[0]-0.5)*3
        params1['PX0']    = sigmapx*(np.random.rand(1)[0]-0.5)*3
        params1['Y0']     = sigmay*(np.random.rand(1)[0]-0.5)*3
        params1['PY0']    = sigmapy*(np.random.rand(1)[0]-0.5)*3
        
        params1['emitX']  = params['emitX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaX']  = params['betaX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaX'] = params['alphaX']*(-1 + 3.0*(np.random.rand(1)[0]) )

        params1['emitY']  = params['emitY']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaY']  = params['betaY']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaY'] = params['alphaY']*(-1 + 3.0*(np.random.rand(1)[0]) )

        params1['Nparticles'] = npart1
        
        psCoords1 = PhaseSpaceSimple4D(params1)   

        params1['X0']     = sigmax*(np.random.rand(1)[0]-0.5)*3
        params1['PX0']    = sigmapx*(np.random.rand(1)[0]-0.5)*3
        params1['Y0']     = sigmay*(np.random.rand(1)[0]-0.5)*3
        params1['PY0']    = sigmapy*(np.random.rand(1)[0]-0.5)*3
        
        params1['emitX']  = params['emitX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaX']  = params['betaX']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaX'] = params['alphaX']*(-1 + 3.0*(np.random.rand(1)[0]) )
        
        params1['emitY']  = params['emitY']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['betaY']  = params['betaY']*(0.2 + 1.6*(np.random.rand(1)[0]) )
        params1['alphaY'] = params['alphaY']*(-1 + 3.0*(np.random.rand(1)[0]) )

        params1['Nparticles'] = npart2
        
        psCoords2 = PhaseSpaceSimple4D(params1)

        phaseSpaceCoords = np.concatenate((psCoords1,psCoords2),axis=1)

        Xdata[i*scanSteps:(i+1)*scanSteps, :], Ydata[i*scanSteps:(i+1)*scanSteps, :] = Sinogram4D(params, phaseSpaceCoords)
        
        PhaseSpaceDensityX, xedges1, xedges2 = np.histogram2d(phaseSpaceCoords[0,:], phaseSpaceCoords[1,:], bins = [xbins,pxbins])

        PhaseSpaceDensityY, yedges1, yedges2 = np.histogram2d(phaseSpaceCoords[2,:], phaseSpaceCoords[3,:], bins = [ybins,pybins])

        Xdata2[i*psresolution:(i+1)*psresolution, :] = PhaseSpaceDensityX

        Ydata2[i*psresolution:(i+1)*psresolution, :] = PhaseSpaceDensityY
        
    print(' Done!')
        
    Xdata = Xdata*1000
    
    Ydata = Ydata*1000
    
    Xdata = Xdata.round(decimals=0)

    Ydata = Ydata.round(decimals=0)

    tag = params['Tag']

    np.savetxt('SinogramsComplexPSX' + tag + '.txt', Xdata, delimiter=',')

    np.savetxt('PhaseSpaceDensityX' + tag + '.txt', Xdata2, delimiter=',')

    np.savetxt('SinogramsComplexPSY' + tag + '.txt', Ydata, delimiter=',')

    np.savetxt('PhaseSpaceDensityY' + tag + '.txt', Ydata2, delimiter=',')

    return Xdata, Xdata2, Ydata, Ydata2



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
    