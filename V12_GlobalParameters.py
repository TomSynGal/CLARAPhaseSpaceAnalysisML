GlobalParams = {'AlphaX': -0.2, 
                'BetaX': 1.5, 
                'EpsilonX': 1.0,
                'PhaseResolution': 48, 
                'Projections': 24, 
                'TomographyProjections': 48, 
                'Samples': 6000,
                'Nparticles': 100000}


import numpy as np
StaticPhaseRange = 3*np.sqrt(GlobalParams['BetaX']*GlobalParams['EpsilonX'])