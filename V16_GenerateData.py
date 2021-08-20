import numpy as np
from V16_Functions import MakeTrainingImages, MakeTomographyTrainingImages
from V16_MacroFunctions import MakeRealImages, MakeRealTomographyImages
from V16_GlobalParameters import GlobalParams

GenerateQuadScanData = False

GenerateTomographyData = False

GenerateRealQuadScanData = True

GenerateRealTomographyData = True


if GenerateQuadScanData:
    
    sinograms, opticsParams = MakeTrainingImages(GlobalParams['AlphaX_Rec'], 
                                                 GlobalParams['BetaX_Rec'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'])
    
    
if GenerateTomographyData:
        
    sinograms_T, phaseSpace_T = MakeTomographyTrainingImages(GlobalParams['AlphaX_Rec'], 
                                                 GlobalParams['BetaX_Rec'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['TomographyProjections'],
                                                 GlobalParams['TomographySamples'])


if GenerateRealQuadScanData:
    
    sinograms_Real, opticsParams_Real = MakeRealImages(GlobalParams['AlphaX_Rec'], 
                                                 GlobalParams['BetaX_Rec'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'],
                                                 GlobalParams['Samples'],
                                                 GlobalParams['Nparticles'])
    
if GenerateRealTomographyData:
    
    sinograms_T_Real, opticsParams_T_Real = MakeRealTomographyImages(GlobalParams['AlphaX_Rec'], 
                                                 GlobalParams['BetaX_Rec'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'],
                                                 GlobalParams['TomographySamples'],
                                                 GlobalParams['Nparticles'])