import numpy as np

tm = np.loadtxt('TransferMatrix-ReconstructionPoint-ObservationPoint.txt')

tmX = np.delete(tm, [2,3], 1)

TransferMatrixX = tmX[~np.all(tmX == 0, axis = 1)]

scanSteps = int(TransferMatrixX.shape[0]/2)

tmY = np.delete(tm, [0,1], 1)

TransferMatrixY = tmY[~np.all(tmY == 0, axis = 1)]

Params = {'X0': 0.0,
                'PX0': 0.0,
                'alphaX': -3.62, #-0.2 -3.620
                'betaX': 26.551, #1.5 26.551
                'emitX': 9.5E-09, #1.0 9.5E-09
                'Y0': 0.0,
                'PY0': 0.0,
                'alphaY': -3.62, #-0.2 -3.620
                'betaY': 26.551, #1.5 26.551
                'emitY': 9.5E-09, #1.0 9.5E-09
                'PhaseSpaceResolution': 48,
                'PhaseSpaceRange': 3,
                'ScanSteps': scanSteps,
                'Samples': 2000,
                'TestSamples': 200,
                'Nparticles': 20000,
                'ComplexPhaseSpace': False,
                'sigmaXObsMax': 0,
                'sigmaYObsMax': 0,
                'PhaseSpaceRangeObs': 5,
                'Tag': '_Test1'}

gammaX = (1 + np.square(Params['alphaX'])) / Params['betaX']

gammaY = (1 + np.square(Params['alphaY'])) / Params['betaY']

SigmaMatxX = np.array([[Params['betaX'], -Params['alphaX']],[-Params['alphaX'], gammaX]])

SigmaMatxY = np.array([[Params['betaY'], -Params['alphaY']],[-Params['alphaY'], gammaY]])

betaXObs = np.zeros((scanSteps, 1))

betaYObs = np.zeros((scanSteps, 1))

for i in range (scanSteps):

    j = 2*i
    k = j+2

    tmX = TransferMatrixX[j:k, :]

    tmXTranspose = np.transpose(tmX)
    
    SigmaMatxXObs = np.matmul(tmX, SigmaMatxX)
    
    SigmaMatxXObs = np.matmul(SigmaMatxXObs, tmXTranspose)
    
    betaXObs[i] = SigmaMatxXObs[0,0]

for i in range (scanSteps):

    j = 2*i
    k = j+2

    tmY = TransferMatrixY[j:k, :]

    tmYTranspose = np.transpose(tmY)
    
    SigmaMatxYObs = np.matmul(tmY, SigmaMatxY)
    
    SigmaMatxYObs = np.matmul(SigmaMatxYObs, tmYTranspose)
    
    betaYObs[i] = SigmaMatxYObs[0,0]

Params['sigmaXObsMax'] = np.sqrt(np.max(betaXObs)*Params['emitX'])

Params['sigmaYObsMax'] = np.sqrt(np.max(betaYObs)*Params['emitY'])
