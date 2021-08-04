# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:02:57 2021

@author: thoma
"""
##############################################################################
#01/XX Brief
##############################################################################
'''

The purpose of this code is to generate data for emittance and optics 
measurements on CLARA, the compact linear accelerator at Daresbury lab.

Prior to this code, data creation occoured in MATLAB. In this study I will
transfer the same mathematics over to Python explainig steps within the code
as I go.

To run this code correctly, packages such as Numpy,Pandas, MatPlotLib, 
Tensorflow and Keras should be installed.

Also there is an accompanying script that contains all of the functions to run
this script.

This is the main script, all tasks should be performed from here, however they
do rely on the functions to run properly.

'''
##############################################################################
#End 01/XX Brief
##############################################################################
#
#
#
##############################################################################
#02/XX Imports
##############################################################################

import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import keras
from PythonFunctionsV01 import MakePhaseSpace, MakeSinogram

##############################################################################
#End 02/XX Imports
##############################################################################
#
#
#
##############################################################################
#03/XX True/False Switchboard
##############################################################################

##############################################################################
#End 03/XX True/False Switchboard
##############################################################################
#
#
#
##############################################################################
#04/XX Global Parameters
##############################################################################

alphaX = -0.2

betaX = 1.5

epsilonX = 1.0

phaseResolution = 48

projections = 24

samples = 6000

##############################################################################
#End 04/XX Global Parameters
##############################################################################


phaseSpace = MakePhaseSpace(alphaX, betaX, epsilonX, phaseResolution)

phaseSpaceXProjection = MakeSinogram(alphaX, betaX, epsilonX, phaseResolution, projections)

