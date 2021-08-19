import numpy as np
from V16_GlobalParameters import GlobalParams, TransformMatrix
Projections = TransformMatrix.shape[0]/2
Projections = int(Projections)
    
gX = (1 + np.square(GlobalParams['AlphaX'])) / GlobalParams['BetaX']

ReconMatx = np.array([[GlobalParams['BetaX'], -GlobalParams['AlphaX']],[-GlobalParams['AlphaX'], gX]])

BetaObs = np.zeros((Projections, 1))

for i in range (Projections):
    j = 2*i
    k = j+2

    MiniMatx = TransformMatrix[j:k, :]
    MiniMatxTranspose = np.transpose(MiniMatx)
    Matx = np.matmul(ReconMatx, MiniMatxTranspose)
    ObsMatx = np.matmul(MiniMatx, Matx)
    BetaObs[i] = ObsMatx[0,0]

BetaObsMax = np.max(BetaObs)
