import numpy as np

Matrix = np.loadtxt('Matrix.txt')
Matrix = np.delete(Matrix, [2,3], 1)
TransformMatrix = Matrix[~np.all(Matrix == 0, axis = 1)]
RealProjections = int(TransformMatrix.shape[0]/2)

GlobalParams = {'AlphaX_Rec': -3.62, #-0.2 -3.620
                'BetaX_Rec': 26.551, #1.5 26.551
                'EpsilonX': 9.5E-09, #1.0 9.5E-09
                'PhaseResolution': 48,
                'Projections': 24, 
                'TomographyProjections': 48, 
                'Samples': 6000,
                'TomographySamples': 3500,
                'Nparticles': 100000}

GlobalParams['PhaseRange_Rec'] = 3*np.sqrt(GlobalParams['BetaX_Rec']*GlobalParams['EpsilonX'])
GlobalParams['RealProjections'] = RealProjections


gX_Rec = (1 + np.square(GlobalParams['AlphaX_Rec'])) / GlobalParams['BetaX_Rec']

ReconMatx = np.array([[GlobalParams['BetaX_Rec'], -GlobalParams['AlphaX_Rec']],[-GlobalParams['AlphaX_Rec'], gX_Rec]])

BetaObs = np.zeros((RealProjections, 1))

for i in range (RealProjections):
    j = 2*i
    k = j+2

    MiniMatx = TransformMatrix[j:k, :]
    MiniMatxTranspose = np.transpose(MiniMatx)
    Matx = np.matmul(ReconMatx, MiniMatxTranspose)
    ObsMatx = np.matmul(MiniMatx, Matx)
    BetaObs[i] = ObsMatx[0,0]

GlobalParams['BetaXMax_Obs'] = np.max(BetaObs)
GlobalParams['PhaseRangeMax_Obs'] = 3*np.sqrt(GlobalParams['BetaXMax_Obs']*GlobalParams['EpsilonX'])
