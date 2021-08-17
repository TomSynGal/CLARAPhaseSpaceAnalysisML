import numpy as np

GlobalParams = {'AlphaX': -0.2, #-0.2 -3.620
                'BetaX': 1.5, #1.5 26.551
                'EpsilonX': 1.0, #1.0 9.5E-09
                'PhaseResolution': 48, 
                'Projections': 24, 
                'TomographyProjections': 48, 
                'Samples': 6000,
                'Nparticles': 100000}

StaticPhaseRange = 3*np.sqrt(GlobalParams['BetaX']*GlobalParams['EpsilonX'])

Matrix = np.loadtxt('Matrix.txt')
Matrix = np.delete(Matrix, [2,3], 1)


TransformMatrix = Matrix[~np.all(Matrix == 0, axis = 1)]