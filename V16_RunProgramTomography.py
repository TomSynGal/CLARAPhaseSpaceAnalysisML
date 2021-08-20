import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from V16_Functions import MultipleHeatMapPlot
from V16_GlobalParameters import GlobalParams

testSamplesT = 500

Use_Real_Simulation_Data = True

if Use_Real_Simulation_Data:
        sinograms_T = np.loadtxt('XdataTomographyReal.txt',delimiter=',')
        phaseSpace_T = np.loadtxt('YdataTomographyReal.txt',delimiter=',')
        nimage_T = phaseSpace_T.shape[0];
        Projections = GlobalParams['RealProjections']
else:
        sinograms_T = np.loadtxt('XdataTomography.txt',delimiter=',')
        phaseSpace_T = np.loadtxt('YdataTomography.txt',delimiter=',')
        nimage_T = phaseSpace_T.shape[0];
        Projections = GlobalParams['TomographyProjections']

sinograms_T = sinograms_T[:,np.newaxis,:,np.newaxis].reshape((nimage_T,Projections,-1,1));
Res = GlobalParams['PhaseResolution']
ResSquare = np.square(Res)
print(sinograms_T.shape)
print(phaseSpace_T.shape)

ratio = Projections/Res


fig = plt.figure(figsize=(16,(5*ratio)))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(sinograms_T[i], interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    
fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(phaseSpace_T[i,:].reshape(Res,Res), interpolation='nearest')
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
    layers.Dense(ResSquare,activation='linear')
])


test_set = 10 + np.arange(6);
mlpredict = linear_model.predict(x_test_T[test_set])
fig = plt.figure(figsize=(16,5))
for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(y_test_T[test_set[0]+i,:].reshape(Res,Res), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(Res,Res), interpolation='nearest')
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
    sub.imshow(y_test_T[test_set[0]+i,:].reshape(Res,Res), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(Res,Res), interpolation='nearest')
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
    sub.imshow(y_test_T[test_set[0]+i,:].reshape(Res,Res), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(Res,Res), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
