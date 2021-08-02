# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 20:17:58 2021

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
#01/14 Import Section
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

#Keras data pre-processing is a tool needed to train a model from raw data
#when using Keras.

##############################################################################
#End 01/14 Import Section
##############################################################################
#
#
#
##############################################################################
#02/14 Data Load and Reshape
##############################################################################

phaseSpace = np.loadtxt('TrainingDataYTomography.txt',delimiter=',')
sinograms = np.loadtxt('TrainingDataXTomography.txt',delimiter=',');

#Data created in MATLAB is compiled in a text file and loaded in using Numpys
#load.txt function where the boundary between characters is specified by the
#delimiter. Two variables are created as Numpy arrays from the data.

nimage = phaseSpace.shape[0];

#Created a new variable of size 1 with a value of 3500, the same size as the
#amount of samples in the opticsParams variable.

sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,48,-1,1));

#Reshapes the sinograms array from a 1x2 array of value (168000,48) to a 1x4
#array of value (3500,48,48,1) user specified and now in line also with the
#size of nimage and thus the sample size of the opticsParams variable.

print(sinograms.shape)
print(phaseSpace.shape)

#Debug code to print the shapes of the Numpy arrays.

##############################################################################
#End 02/14 Data Load and Reshape
##############################################################################
#
#
#
##############################################################################
#03/14 Figure Plots
##############################################################################

fig = plt.figure(figsize=(16,5))

#Defines a figure plot variable with user specified dimensions of 16x5.

for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(sinograms[i], interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
#A for loop with a length of 24 repetitions that generates 24 sub plots in to
#a single figure with dimensions of 8x3 showing the sinograms data. This data
#is displayed as a kind of heat map where the brighter parts of the image
#denote a higher density.

#These graphs are smeared in appearance as they are yet to be stitched together
#to form a useful image of the beam.

##############################################################################
#End 03/14 Figure Plots
##############################################################################
#
#
#
##############################################################################
#04/14 Figure Plots Continued
##############################################################################

fig = plt.figure(figsize=(16,5))

#Defines another figure plot variable with user specified dimensions of 16x5.

for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(phaseSpace[i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
#The same 24 repetition for loop this time used for the phaseSpace dataset.

#These graphs show the fully analysied versions of the previous figures
#computationally stitched together to accuratley reprisent the cross section
#of the beam in the x-axis of phase space.

##############################################################################
#End 04/14 Figure Plots Continued
############################################################################## 
#
#
#
##############################################################################
#05/14 The Test-Train Split
##############################################################################

#The Test-Train split is a crucial part of the machine learning process,
#removing a chunk of unbiased data as a test sample that is independent of the
#network to use later to test the networks performance.

n_test = 500;

#Defines the amount of samples in the test dataset, 500 in this case, leaving
#3000 for training.

psScale = 1000;

#New variable psScale with a value of 1000. 

x_test = sinograms[:n_test,:,:,:];

#New variable x_test geerated from the first 500 samples of the variable
#sinograms.


y_test = phaseSpace[:n_test,:]/psScale;

#Same procedure for y_test which is divided by psScale.

x_train = sinograms[n_test:,:,:,:];
y_train = phaseSpace[n_test:,:]/psScale;

#The two training samples recieve the same procedure also but for the final
#3000 of the samples.

print(x_test.shape)
print(x_train.shape)

#Debug code printing the shapes of the x_test and x_train arrays.

##############################################################################
#End 05/14 The Test-Train Split
##############################################################################
#
#
#
##############################################################################
#06/14 Model Creation
##############################################################################

#Creation of the neural network architecture.

linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=x_test.shape[1:]),
    layers.Flatten(),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(2304,activation='linear')
])

#BatchNormalization transforms the data to a mean close to zero and a standard
#deviation close to 1. The input shape is also specified.

#layers.Flatten, flattens the input without affecting the batch size.

#3 300 neuron layers dense layers are created. Dense layers are fully
#connected wuith each neuron connected to all of the previous and following
#neurons.

#A final 2304 neuron layer to generate the desired output.

#Linear activation function reprisented by the line y=x that intersects at
#the origin.

##############################################################################
#End 06/14 Mocel Creation
##############################################################################
#
#
#
##############################################################################
#07/14 Initial Figure Creation
##############################################################################

test_set = 10 + np.arange(6);

#np.arange integers from 0 to 5.

mlpredict = linear_model.predict(x_test[test_set])

#Forming initial predictions on the x-axis test data set

fig = plt.figure(figsize=(16,5))

#Make a plot with a 16x5 aspect ratio.

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(y_test[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

#A for loop for a 12 sub-plot figure in a 2x6 arrangment.

#The plots compare the initial predictions from the actual phase space images.

##############################################################################
#End 07/14 Initial Figure Creation
##############################################################################
#
#
#
##############################################################################
#08/14 Model Compiler
##############################################################################

#The moedel.compile function to configure the model for training with the
#adam optimiser and a user defined learning rate and loss function.

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_absolute_error')

##############################################################################
#End 08/14 Model Compiler
##############################################################################
#
#
#
##############################################################################
#09/14 Running The Model
##############################################################################

#The model.fit function begins the training of the model. 

#The number of itterations is set to 30 with a batch size per epoch of 20
#which should yield a quick training cycle.

history = linear_model.fit(
    x_train, y_train, 
    epochs=30,
    batch_size=20,
    # suppress logging
    verbose=1,
    
#Verbose=1 to show progress animations without excessive logging readout.
    
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

#20% validation split so 600 units of the 3000 large train data set is used
#to validate this supervised learning neural network. This split is optimal as
#the test data has already been removed prior to this split essentially
#yielding two independent validations from which to monitor the networks
#performance.

##############################################################################
#End 09/14 Running the Model
##############################################################################
#
#
#
##############################################################################
#10/14 Post Model Imaging
##############################################################################

test_set = 50 + np.arange(6);

#Using np.arrange to take integer values from 50 to 55.

mlpredict = linear_model.predict(x_test[test_set])

#Repeating predictions now the model has been trained.

fig = plt.figure(figsize=(16,5))

#Make a plot with a 16x5 aspect ratio.

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(y_test[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
#The foor loop again generates 12 images in 1 figure in a 2x6 arrangment.
#This time after training the lower images don't look like noise.
#They now strongly resemble the original sinograms made in MATLAB.

##############################################################################
#End 10/14 Post Model Imaging
##############################################################################
#
#
#
##############################################################################
#11/14 Second Model Compiler
##############################################################################

#A second moedel.compile function to configure the model for training with the
#adam optimiser and a user defined learning rate and loss function.

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_squared_error')

##############################################################################
#End 11/14 Second Model Compiler
##############################################################################
#
#
#
##############################################################################
#12/14 Running The Second Model
##############################################################################

#The model.fit function begins the training of the second model. 

#The number of itterations is set to 20 with a batch size per epoch of 20
#which should yield an even faster training cycle than the previous network.

#Most likely 2 networks is this code to test the justification for 10 extra
#epochs in the first model to conserve computational power.

history = linear_model.fit(
    x_train, y_train, 
    epochs=20,
    batch_size=20,
    # suppress logging
    verbose=1,
    
#Verbose=1 to show progress animations without excessive logging readout.

    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

#20% validation split so 600 units of the 3000 large train data set is used
#to validate this supervised learning neural network. This split is optimal as
#the test data has already been removed prior to this split essentially
#yielding two independent validations from which to monitor the networks
#performance.

##############################################################################
#End 12/14 Running The Second Model
##############################################################################
#
#
#
##############################################################################
#13/14 Second Model Post Imaging
##############################################################################

test_set = 100 + np.arange(6);

#Using np.arrange to take integer values from 100 to 105 so testing different
#samples from earlier.

mlpredict = linear_model.predict(x_test[test_set])
fig = plt.figure(figsize=(16,5))

#Make a plot with a 16x5 aspect ratio.

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(y_test[test_set[0]+i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(48,48), interpolation='nearest')
    plt.xticks([])
    
#The foor loop again generates 12 images in 1 figure in a 2x6 arrangment.
#This time after training the lower images don't look like noise.
#They now strongly resemble the original sinograms made in MATLAB.

#These plots can be compared and contrasted to the plots from the previous
#figure to see if there are more artefacts in a figure generated through
#20 cycles as opposed to 30.

##############################################################################
#End 13/14 Second Model Post Imaging
##############################################################################
#
#
#
##############################################################################
#14/14 Notes
##############################################################################
'''
Text
'''
##############################################################################
#End 14/14 Notes
##############################################################################