##############################################################################
from V09_GlobalFunctions import MakeTrainingImages
from V09_GlobalFunctions import MakeTomographyTrainingImages
from V09_GlobalFunctions import MakeMacroTrainingImages
from V09_GlobalFunctions import MakeMacroMATXTrainingImages
##############################################################################

GenerateQuadScanData = False

GenerateTomographyData = False

GenerateMacroQuadPhase1 = False

GenerateMacroMATXPhase2 = True

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
#
#
#
##############################################################################
# Macro Matrix Phase 2, MP2
##############################################################################
alphaXMP2 = -0.2

betaXMP2 = 1.5

epsilonXMP2 = 1.0#E-06

phaseResolutionMP2 = 48

projectionsMP2 = 24

samplesMP2 = 6000

NparticlesMP2 = 100000
##############################################################################

if GenerateMacroMATXPhase2:
    
    sinogramsMP1, opticsParamsMP1 = MakeMacroMATXTrainingImages(alphaXMP2,betaXMP2,
                                                            epsilonXMP2,
                                                            phaseResolutionMP2,
                                                            projectionsMP2,
                                                            samplesMP2,
                                                            NparticlesMP2)
##############################################################################
#
##############################################################################