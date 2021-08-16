import numpy as np
from V12_Functions import MakeTrainingImages, MakeTomographyTrainingImages
from V12_MacroFunctions import MakeTrainingImagesPhase1
from V12_MacroFunctions import MakeTrainingImagesPhase2
from V12_MacroFunctions import MakeTrainingImagesPhase3
from V12_GlobalParameters import GlobalParams

GenerateQuadScanData = False

GenerateTomographyData = False

GenerateQuadScan_Realistic_Phase1 = True

GenerateQuadScan_Realistic_Phase2 = True

GenerateQuadScan_Realistic_Phase3 = False


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

if GenerateQuadScan_Realistic_Phase1:
    
    sinogramsMP1, opticsParamsMP1 = MakeTrainingImagesPhase1(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'],
                                                 GlobalParams['Nparticles'])

if GenerateQuadScan_Realistic_Phase2:
    
    sinogramsMP2, opticsParamsMP2 = MakeTrainingImagesPhase2(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'],
                                                 GlobalParams['Nparticles'])

if GenerateQuadScan_Realistic_Phase3:
    
    sinogramsMP3, opticsParamsMP3 = MakeTrainingImagesPhase3(GlobalParams['AlphaX'], 
                                                 GlobalParams['BetaX'], 
                                                 GlobalParams['EpsilonX'], 
                                                 GlobalParams['PhaseResolution'], 
                                                 GlobalParams['Projections'],
                                                 GlobalParams['Samples'],
                                                 GlobalParams['Nparticles'])