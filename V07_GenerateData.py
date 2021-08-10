##############################################################################
from V07_GlobalFunctions import MakeTrainingImages, MakeTomographyTrainingImages
##############################################################################

GenerateQuadScanData = True

GenerateTomographyData = True

##############################################################################
alphaX = -0.2

betaX = 1.5

epsilonX = 1.0#E-06

phaseResolution = 48

projections = 24

samples = 6000
##############################################################################
if GenerateQuadScanData:
    
    sinograms, opticsParams = MakeTrainingImages(alphaX, betaX, epsilonX, 
                                                 phaseResolution, projections,
                                                 samples)
##############################################################################
#
##############################################################################
#
#
#
##############################################################################
#
##############################################################################
alphaXT = -0.2

betaXT = 1.5

epsilonXT = 1.0#E-06

phaseResolutionT = 48

projectionsT = 48

samplesT = 3500
##############################################################################
if GenerateTomographyData:
        
    sinograms_T, phaseSpace_T = MakeTomographyTrainingImages(alphaXT, betaXT, 
                                                    epsilonXT, 
                                                    phaseResolutionT, 
                                                    projectionsT, samplesT)
##############################################################################
#
##############################################################################