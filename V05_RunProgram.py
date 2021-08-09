##############################################################################
# 01/XX Brief.
##############################################################################
'''
The purpose of this code is to generate data for emittance and optics 
measurements on CLARA, the compact linear accelerator at Daresbury lab.

Prior to this code, data creation occoured in MATLAB. In this study I will
transfer the same mathematics over to Python explainig steps within the code
as I go.

To run this code correctly, packages such as Numpy, Pandas, MatPlotLib, 
Tensorflow and Keras should be installed.

Also there is an accompanying script that contains all of the functions to run
this script.

This is the main script, all tasks should be performed from here, however they
do rely on the functions to run properly.
'''
##############################################################################
# 01/XX End, Brief.
##############################################################################
#
#
#
##############################################################################
# 02/XX Imports.
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from V05_Functions import MakePhaseSpace, MakeSinogram, ArrayHeatMapPlot
from V05_Functions import MakeTrainingImages, MultipleHeatMapPlot
##############################################################################
# 02/XX End, Imports.
##############################################################################
#
#
#
##############################################################################
# 03/XX True/False Switchboard.
##############################################################################

GenerateNewData = False

GenerateDataPlots = False

RunProgram = False

##############################################################################
# 03/XX End, True/False Switchboard.
##############################################################################
#
#
#
##############################################################################
# 04/XX Global Parameters.
##############################################################################

alphaX = -0.2

betaX = 1.5

epsilonX = 1.0#E-06

phaseResolution = 48

projections = 24

samples = 6000

testSamples = 1000 #(Must be less than samples)

##############################################################################
# 04/XX End, Global Parameters.
##############################################################################
#
#
#
##############################################################################
# 05/XX Generate New Training Data.
##############################################################################

if GenerateNewData:
    
    Xdata, Ydata = MakeTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                                      projections, samples)
    
else:
    Xdata = sinograms = np.loadtxt('Xdata.txt',delimiter=',')
    Ydata = opticsParams = np.loadtxt('Ydata.txt',delimiter=',')

nimage = opticsParams.shape[0];
sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,24,-1,1));
print(sinograms.shape)
print(opticsParams.shape)

##############################################################################
# 05/XX End, Generate New Training Data.
##############################################################################
#
#
#
##############################################################################
# 06/XX Pre-Network Plots.
##############################################################################

if GenerateDataPlots:
    
    figure = MultipleHeatMapPlot(16, 5, sinograms, 'sinograms')

##############################################################################
# 06/XX End, Pre-Network Plots.
##############################################################################
#
#
#
##############################################################################
# 07/XX Run Program.
##############################################################################

if RunProgram:
    
    
    opticsScale = np.diag([1.0, 1.0, -5.0])
    print(opticsScale)
    n_test = testSamples;
    x_test = sinograms[:n_test,:,:,:];
    print(x_test.shape)
    y_test = np.matmul(opticsParams[:n_test,:],opticsScale);
    x_train = sinograms[n_test:,:,:,:];
    print(x_train.shape)
    y_train = np.matmul(opticsParams[n_test:,:],opticsScale);
    
    linear_model = tf.keras.Sequential([
        layers.BatchNormalization(input_shape=x_test.shape[1:]),
        layers.Flatten(),
        layers.Dense(200),
        layers.Dense(200),
        layers.Dense(200),
        layers.Dense(3,activation='linear')
    ])
    
    mlpredict0 = linear_model.predict(x_test)
    #plt.plot(y_test[:,0],mlpredict0[:,0],'.');
    #plt.plot(y_test[:,1],mlpredict0[:,1],'.');
    #plt.plot(y_test[:,2],mlpredict0[:,2],'.');
    
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
        loss='mean_squared_error')
    
    history = linear_model.fit(
        x_train, y_train, 
        epochs=30,
        batch_size=50,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2)
    
    mlpredict = linear_model.predict(x_test)
    fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(14,8))
    
    ax[0,0].plot(y_test[:,0],mlpredict[:,0],'.');
    ax[0,0].set_xlabel('True emittance');
    ax[0,0].set_ylabel('Fitted emittance');
    
    ax[0,1].plot(y_test[:,1],mlpredict[:,1],'.');
    ax[0,1].set_xlabel('True beta');
    ax[0,1].set_ylabel('Fitted beta');
    
    ax[0,2].plot(y_test[:,2],mlpredict[:,2],'.');
    ax[0,2].set_xlabel('True alpha');
    ax[0,2].set_ylabel('Fitted alpha');
    
    ratioEmittance = np.divide(y_test[:,0],mlpredict[:,0]);
    ax[1,0].hist(ratioEmittance,bins=np.linspace(0.8,1.2,num=20));
    ax[1,0].set_xlabel('True/fitted emittance');
    ax[1,0].set_ylabel('Number of cases');
    
    ratioBeta = np.divide(y_test[:,1],mlpredict[:,1]);
    ax[1,1].hist(ratioBeta,bins=np.linspace(0.8,1.2,num=20));
    ax[1,1].set_xlabel('True/fitted beta');
    ax[1,1].set_ylabel('Number of cases');
    
    ratioAlpha = np.divide(y_test[:,2],mlpredict[:,2]);
    ax[1,2].hist(ratioAlpha,bins=np.linspace(0.5,1.5,num=20));
    ax[1,2].set_xlabel('True/fitted alpha');
    ax[1,2].set_ylabel('Number of cases');
    
    print(f'Mean of true/fitted emittance = {np.mean(ratioEmittance):.3f}')
    print(f'Standard deviation of true/fitted emittance = {100*np.std(ratioEmittance):.2f}%')
    print()
    print(f'Mean of true/fitted beta = {np.mean(ratioBeta):.3f}')
    print(f'Standard deviation of true/fitted beta = {100*np.std(ratioBeta):.2f}%')
    print()
    print(f'Mean of true/fitted alpha = {np.mean(ratioAlpha):.3f}')
    print(f'Standard deviation of true/fitted alpha = {100*np.std(ratioAlpha):.2f}%')

##############################################################################
# 07/XX End, Run Program.
##############################################################################