import numpy as np
from V15_Functions import MakeTrainingImages, MakeTomographyTrainingImages
from V15_MacroFunctions import MakeRealImages, MakeRealTomographyImages
from V15_GlobalParameters import GlobalParams

GenerateQuadScanData = False

GenerateTomographyData = False

GenerateRealQuadScanData = False

GenerateRealTomographyData = True


if GenerateQuadScanData:
    
    sinograms, opticsParams = MakeTrainingImages(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'])
    
    
if GenerateTomographyData:
        
    sinograms_T, phaseSpace_T = MakeTomographyTrainingImages(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'])


if GenerateRealQuadScanData:
    
    sinograms_Real, opticsParams_Real = MakeRealImages(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'],
                                                 GlobalParams['Samples'],
                                                 GlobalParams['Nparticles'])
    
if GenerateRealTomographyData:
    
    sinograms_T_Real, opticsParams_T_Real = MakeRealTomographyImages(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'],
                                                 GlobalParams['TomographySamples'],
                                                 GlobalParams['Nparticles'])