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
from V06_Functions import MakePhaseSpace, MakeSinogram, ArrayHeatMapPlot
from V06_Functions import MakeTrainingImages, MultipleHeatMapPlot
from V06_Functions import MakeTomographyPhaseSpace, MakeTomographySinogram
from V06_Functions import MakeTomographyTrainingImages, MultipleHeatMapPlot2
##############################################################################
# 02/XX End, Imports.
##############################################################################
#
#
#
##############################################################################
# 03/XX True/False Switchboard.
##############################################################################

DoQuadScan = False

GenerateNewData = False

GenerateDataPlots = False

RunProgram = False

DoTomography = False ###

GenerateNewTomographyData = False ###

RunTomographyProgram = False ###

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

alphaXT = -0.2

betaXT = 1.5

epsilonXT = 1.0#E-06

phaseResolutionT = 48

projectionsT = 48

samplesT = 3500

testSamplesT = 500 #(Must be less than samples)

##############################################################################
# 04/XX End, Global Parameters.
##############################################################################
#
#
#
##############################################################################
# XX/XX Generate New Training Data.
##############################################################################

if DoQuadScan:

    if GenerateNewData:
        
        sinograms, opticsParams = MakeTrainingImages(alphaX, betaX, epsilonX, phaseResolution,
                                          projections, samples)
    
        nimage = opticsParams.shape[0];
        sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,24,-1,1));
        print(sinograms.shape)
        print(opticsParams.shape)
    
    else:
        sinograms = np.loadtxt('Xdata.txt',delimiter=',')
        opticsParams = np.loadtxt('Ydata.txt',delimiter=',')
        nimage = opticsParams.shape[0];
        sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,24,-1,1));
        print(sinograms.shape)
        print(opticsParams.shape)

##############################################################################
# XX/XX End, Generate New Training Data.
##############################################################################
#
#
#
##############################################################################
# XX/XX Generate New Tomography Training Data.
##############################################################################

if DoTomography:

    if GenerateNewTomographyData:
        
        sinograms_T, phaseSpace_T = MakeTomographyTrainingImages(alphaXT, betaXT, 
                                                        epsilonXT, 
                                                        phaseResolutionT, 
                                                        projectionsT, samplesT)
        
        nimage_T = phaseSpace_T.shape[0];
        sinograms_T = sinograms_T[:,np.newaxis,:,np.newaxis].reshape((nimage_T,48,-1,1));
        print(sinograms_T.shape)
        print(phaseSpace_T.shape)
    
    
    else:
        sinograms_T = np.loadtxt('XdataTomography.txt',delimiter=',')
        phaseSpace_T = np.loadtxt('YdataTomography.txt',delimiter=',')
        nimage_T = phaseSpace_T.shape[0];
        sinograms_T = sinograms_T[:,np.newaxis,:,np.newaxis].reshape((nimage_T,48,-1,1));
        print(sinograms_T.shape)
        print(phaseSpace_T.shape)

##############################################################################
# XX/XX End, Generate New Tomography Training Data.
##############################################################################
#
#
#
##############################################################################
# XX/XX Pre-Network Plots.
##############################################################################

if GenerateDataPlots:
    
    figure = MultipleHeatMapPlot(16, 5, sinograms, 'sinograms')

##############################################################################
# XX/XX End, Pre-Network Plots.
##############################################################################
#
#
#
##############################################################################
# XX/XX Run Program.
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
# XX/XX End, Run Program.
##############################################################################
#
#
#
##############################################################################
# XX/XX Run Tomography Program.
##############################################################################

if RunTomographyProgram:

    fig = plt.figure(figsize=(16,5))
    for i in range(24):
        sub = fig.add_subplot(3, 8, i + 1)
        sub.imshow(sinograms_T[i], interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    
    
    fig = plt.figure(figsize=(16,5))
    for i in range(24):
        sub = fig.add_subplot(3, 8, i + 1)
        sub.imshow(phaseSpace_T[i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    
    n_test_T = testSamplesT;
    psScale = 1000;
    x_test_T = sinograms_T[:n_test_T,:,:,:];
    print(x_test_T.shape)
    y_test_T = phaseSpace_T[:n_test_T,:]/psScale;
    x_train_T = sinograms_T[n_test_T:,:,:,:];
    print(x_train_T.shape)
    y_train_T = phaseSpace_T[n_test_T:,:]/psScale;
    
    
    linear_model = tf.keras.Sequential([
        layers.BatchNormalization(input_shape=x_test_T.shape[1:]),
        layers.Flatten(),
        layers.Dense(300),
        layers.Dense(300),
        layers.Dense(300),
        layers.Dense(2304,activation='linear')
    ])
    
    
    test_set = 10 + np.arange(6);
    mlpredict = linear_model.predict(x_test_T[test_set])
    fig = plt.figure(figsize=(16,5))
    for i in range(6):
        sub = fig.add_subplot(2, 6, i + 1)
        sub.imshow(y_test_T[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        sub = fig.add_subplot(2, 6, i + 7)
        sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    
    
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
        loss='mean_absolute_error')
    
    
    history = linear_model.fit(
        x_train_T, y_train_T, 
        epochs=30,
        batch_size=20,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2)
    
    
    test_set = 50 + np.arange(6);
    mlpredict = linear_model.predict(x_test_T[test_set])
    fig = plt.figure(figsize=(16,5))
    for i in range(6):
        sub = fig.add_subplot(2, 6, i + 1)
        sub.imshow(y_test_T[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        sub = fig.add_subplot(2, 6, i + 7)
        sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    
    
    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
        loss='mean_squared_error')
    
    
    history = linear_model.fit(
        x_train_T, y_train_T, 
        epochs=20,
        batch_size=20,
        # suppress logging
        verbose=1,
        # Calculate validation results on 20% of the training data
        validation_split = 0.2)
    
    
    test_set = 100 + np.arange(6);
    mlpredict = linear_model.predict(x_test_T[test_set])
    fig = plt.figure(figsize=(16,5))
    for i in range(6):
        sub = fig.add_subplot(2, 6, i + 1)
        sub.imshow(y_test_T[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        sub = fig.add_subplot(2, 6, i + 7)
        sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

##############################################################################
# XX/XX End, Run Tomography Program.
##############################################################################

