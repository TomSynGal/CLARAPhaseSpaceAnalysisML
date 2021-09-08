import numpy as np

tm = np.loadtxt('TransferMatrix-ReconstructionPoint-ObservationPoint.txt')

tm = np.delete(tm, [2,3], 1)

TransferMatrix = tm[~np.all(tm == 0, axis = 1)]

scanSteps = int(TransferMatrix.shape[0]/2)

GlobalParams = {'X0': 0.0,
                'PX0': 0.0,
                'alphaX': -3.62, #-0.2 -3.620
                'betaX': 26.551, #1.5 26.551
                'emitX': 9.5E-09, #1.0 9.5E-09
                'PhaseSpaceResolution': 48,
                'PhaseSpaceRange': 3,
                'ScanSteps': scanSteps,
                'Samples': 2000,
                'TestSamples': 200,
                'Nparticles': 20000,
                'ComplexPhaseSpace': False,
                'sigmaXObsMax': 0,
                'PhaseSpaceRangeObs': 5,
                'Tag': '_Test1'}

gammaX = (1 + np.square(GlobalParams['alphaX'])) / GlobalParams['betaX']

SigmaMatx = np.array([[GlobalParams['betaX'], -GlobalParams['alphaX']],[-GlobalParams['alphaX'], gammaX]])

betaXObs = np.zeros((scanSteps, 1))

for i in range (scanSteps):

    j = 2*i
    k = j+2

    tm = TransferMatrix[j:k, :]

    tmTranspose = np.transpose(tm)
    
    SigmaMatxObs = np.matmul(tm, SigmaMatx)
    
    SigmaMatxObs = np.matmul(SigmaMatxObs, tmTranspose)
    
    betaXObs[i] = SigmaMatxObs[0,0]

GlobalParams['sigmaXObsMax'] = np.sqrt(np.max(betaXObs)*GlobalParams['emitX'])
