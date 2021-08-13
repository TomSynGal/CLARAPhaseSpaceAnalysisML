##############################################################################
from V08_GlobalFunctions import MakeTrainingImages
from V08_GlobalFunctions import MakeTomographyTrainingImages
from V08_GlobalFunctions import MakeMacroTrainingImages
##############################################################################

GenerateQuadScanData = False

GenerateTomographyData = False

GenerateMacroQuadPhase1 = True

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
#
#
#
##############################################################################
# Macro Phase 1, MP1
##############################################################################
alphaXMP1 = -0.2

betaXMP1 = 1.5

epsilonXMP1 = 1.0#E-06

phaseResolutionMP1 = 48

projectionsMP1 = 24

samplesMP1 = 12000

NparticlesMP1 = 100000
##############################################################################

if GenerateMacroQuadPhase1:
    
    sinogramsMP1, opticsParamsMP1 = MakeMacroTrainingImages(alphaXMP1,betaXMP1,
                                                            epsilonXMP1,
                                                            phaseResolutionMP1,
                                                            projectionsMP1,
                                                            samplesMP1,
                                                            NparticlesMP1)
##############################################################################
#
##############################################################################