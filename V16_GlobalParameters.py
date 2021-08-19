import numpy as np

Matrix = np.loadtxt('Matrix.txt')
Matrix = np.delete(Matrix, [2,3], 1)
TransformMatrix = Matrix[~np.all(Matrix == 0, axis = 1)]
Projections = TransformMatrix.shape[0]/2
Projections = int(Projections)

GlobalParams = {'AlphaX': -3.62, #-0.2 -3.620
                'BetaX': 26.551, #1.5 26.551
                'EpsilonX': 1.0, #1.0 9.5E-09
                'PhaseResolution': 16, #was 48 for quad, tomog and real quad
                'Projections': 24, 
                'TomographyProjections': 48, 
                'Samples': 6000,
                'TomographySamples': 3500,
                'Nparticles': 100000}

GlobalParams['StaticPhaseRange'] = 3*np.sqrt(GlobalParams['BetaX']*GlobalParams['EpsilonX'])
