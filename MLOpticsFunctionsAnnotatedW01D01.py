# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:30:55 2021

@author: thoma
"""

##############################################################################
#Annotation Exercise
#Step 1
#Thomas Gallagher
#Monday 2nd August 2021
##############################################################################
#
#
#
##############################################################################
#01/12 Import Section
##############################################################################

#The import system in python gains access to modules by searching for them
#and then binding them to the code for use later.

import numpy as np

#Numpy used for array and matrix manipulation and also the computation of
#mathematical functions with ease.

from matplotlib import pyplot as plt

#Matplotlib used as a second hand to Numpy to plot and visualise results.
#Pyplot in particular is defined as a collection of functions that make 
#Matplotlib operate in a similar sense to MATLAB.

import tensorflow as tf

#Tensorflow is the chosen machine learning platform for this study.

from tensorflow.keras import layers

#Keras is the deep learning API ("Application Programming Interface") now
#packaged within Tensorflow. The layers import forms the foundation for
#creating a neural network using Keras.

from tensorflow.keras.layers.experimental import preprocessing

#Keras data pre-processing is a tool needed to train a model from raw data
#when using Keras.

##############################################################################
#End 01/12 Import Section
##############################################################################
#
#
#
##############################################################################
#02/12 Data Load and Reshape
##############################################################################

opticsParams = np.loadtxt('TrainingDataY.txt',delimiter=',')
sinograms = np.loadtxt('TrainingDataX.txt',delimiter=',');

#Data created in MATLAB is compiled in a text file and loaded in using Numpys
#load.txt function where the boundary between characters is specified by the
#delimiter. Two variables are created as Numpy arrays from the data.

nimage = opticsParams.shape[0];

#Created a new variable of size 1 with a value of 6000, the same size as the
#amount of samples in the opticsParams variable.

sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,24,-1,1));

#Reshapes the sinograms array from a 1x2 array of value (144000,48) to a 1x4
#array of value (6000,24,48,1) user specified and now in line also with the
#size of nimage and thus the sample size of the opticsParams variable.

print(sinograms.shape)
print(opticsParams.shape)

#Debug code to print the shapes of the Numpy arrays.

##############################################################################
#End 02/12 Data Load and Reshape
##############################################################################
#
#
#
##############################################################################
#03/12 Figure Plots
##############################################################################

fig = plt.figure(figsize=(16,5))

#Defines a figure plot variable with user specified dimensions of 16x5.

for i in range(32):
    sub = fig.add_subplot(4, 8, i + 1)
    sub.imshow(sinograms[i], interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

#A for loop with a length of 32 repetitions that generates 32 sub plots in to
#a single figure with dimensions of 8x4 showing the sinograms data. This data
#is displayed as a kind of heat map where the brighter parts of the image
#denote a higher density.

##############################################################################
#End 03/12 Figure Plots
##############################################################################
#
#
#
##############################################################################
#04/12 Optics Scale
##############################################################################

opticsScale = np.diag([1.0, 1.0, -5.0])

#Defines a new variable opticsScale as a Numpy array and sets its values.

print(opticsScale)

#Debug code to print the Numpy array.

##############################################################################
#End 04/12 Optics Scale
############################################################################## 
#
#
#
##############################################################################
#05/12 The Test-Train Split
##############################################################################

#The Test-Train split is a crucial part of the machine learning process,
#removing a chunk of unbiased data as a test sample that is independent of the
#network to use later to test the networks performance.

n_test = 1000;

#Defines the amount of samples in the test dataset, 1000 in this case, leaving
#5000 for training.

x_test = sinograms[:n_test,:,:,:];

#New variable x_test geerated from the first 1000 samples of the variable
#sinograms.


y_test = np.matmul(opticsParams[:n_test,:],opticsScale);

#Same for y_test as for x_test using the numpy matrix multiplication function
#for the first 1000 samples.

x_train = sinograms[n_test:,:,:,:];
y_train = np.matmul(opticsParams[n_test:,:],opticsScale);

#Same again for the new variables x_train and y_train which both recieve the
#last 5000 samples for training. This affords around 17% of the total data
#to be used at the end to test the network performance.

print(x_test.shape)
print(x_train.shape)

#Debug code printing the shapes of the x_test and x_train arrays.

##############################################################################
#End 05/12 The Test-Train Split
##############################################################################
#
#
#
##############################################################################
#06/12 Model Creation
##############################################################################

#Creation of the neural network architecture.

linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=x_test.shape[1:]),
    layers.Flatten(),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(3,activation='linear')
])

#BatchNormalization transforms the data to a mean close to zero and a standard
#deviation close to 1. The input shape is also specified.

#layers.Flatten, flattens the input without affecting the batch size.

#3 200 neuron layers dense layers are created. Dense layers are fully
#connected wuith each neuron connected to all of the previous and following
#neurons.

#A final 3 neuron layer to focus in on the 3 desired outputs. The beam
#emittance and the alpha and beta Twiss parameters.

#Linear activation function reprisented by the line y=x that intersects at
#the origin.

##############################################################################
#End 06/12 Mocel Creation
##############################################################################
#
#
#
##############################################################################
#07/12
##############################################################################

mlpredict0 = linear_model.predict(x_test)

#Using the model.predict function to generate predictions for the expected
#outputs of the neural network for the 3 desired outputs.

plt.plot(y_test[:,0],mlpredict0[:,0],'.');
plt.plot(y_test[:,1],mlpredict0[:,1],'.');
plt.plot(y_test[:,2],mlpredict0[:,2],'.');

#A plot of these 3 outputs on the same graph. This currently doesn't show in
#Spyder so as I begin to edit I will add a plt.show here to visualaise this
#graph in Spyder.

##############################################################################
#End 07/12
##############################################################################
#
#
#
##############################################################################
#08/12
##############################################################################

#The moedel.compile function to configure the model for training with the
#adam optimiser and a user defined learning rate and loss function.

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_squared_error')

##############################################################################
#End 08/12
##############################################################################
#
#
#
##############################################################################
#09/12
##############################################################################

#The model.fit function begins the training of the model. 

#The number of itterations is set to 30 with a batch size per epoch of 50
#which should yield a quick training cycle.

history = linear_model.fit(
    x_train, y_train, 
    epochs=30,
    batch_size=50,
    # suppress logging
    
#Verbose=1 to show progress animations without excessive logging readout.
    
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

#20% validation split so 1000 units of the 5000 large train data set is used
#to validate this supervised learning neural network. This split is optimal as
#the test data has already been removed prior to this split essentially
#yielding two independent validations from which to monitor the networks
#performance.

##############################################################################
#End 09/12
##############################################################################
#
#
#
##############################################################################
#10/12
##############################################################################

#Code to compile the neural network output into charts and graphs.

mlpredict = linear_model.predict(x_test)
fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(14,8))

#Code that defines the figure as 6 plots with 2 rows and 3 columns with a 14x8
#aspect ratio.

#The first three figure plots show the true values of the 3 neural network 
#outputs as a function of the neural network output using only the test data
#which thus far the network has not seen to determine wether both
#sets of data are in agreement with one another.

#Plot 1 for emittance.

ax[0,0].plot(y_test[:,0],mlpredict[:,0],'.');
ax[0,0].set_xlabel('True emittance');
ax[0,0].set_ylabel('Fitted emittance');

#Plot 2 for the Twiss parameter Beta.

ax[0,1].plot(y_test[:,1],mlpredict[:,1],'.');
ax[0,1].set_xlabel('True beta');
ax[0,1].set_ylabel('Fitted beta');

#Plot 3 for the Twiss parameter Alpha.

ax[0,2].plot(y_test[:,2],mlpredict[:,2],'.');
ax[0,2].set_xlabel('True alpha');
ax[0,2].set_ylabel('Fitted alpha');

#The final 3 plots determine the ratio of the 3 neural network outputs
#against the outputs that the neural network generated using the test sample
#which was not used during network training.

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

##############################################################################
#End 10/12
##############################################################################
#
#
#
##############################################################################
#11/12
##############################################################################

#The final part of the code is comprised of print statements determining the 
#values of the three calculated parameters. Numpys mean and standard deviation
#functions are used to determine the means of these values and their associated
#errors.

print(f'Mean of true/fitted emittance = {np.mean(ratioEmittance):.3f}')
print(f'Standard deviation of true/fitted emittance = {100*np.std(ratioEmittance):.2f}%')
print()
print(f'Mean of true/fitted beta = {np.mean(ratioBeta):.3f}')
print(f'Standard deviation of true/fitted beta = {100*np.std(ratioBeta):.2f}%')
print()
print(f'Mean of true/fitted alpha = {np.mean(ratioAlpha):.3f}')
print(f'Standard deviation of true/fitted alpha = {100*np.std(ratioAlpha):.2f}%')

##############################################################################
#End 11/12
##############################################################################
#
#
#
##############################################################################
#12/12 Notes and Thoughts
##############################################################################
'''
Text
'''
##############################################################################
#End 12/12 Notes and Thoughts
##############################################################################