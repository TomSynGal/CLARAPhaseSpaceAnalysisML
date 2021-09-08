import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from V17_Parameters import GlobalParams
from V17_MakeSinograms import MakeSinograms

GlobalParams['ComplexPhaseSpace'] = True

generateSinograms = True


if generateSinograms:

    sinograms, psdensity = MakeSinograms(GlobalParams)

else:
        
    sinograms = np.loadtxt('SinogramsComplexPS.txt',delimiter=',')
    psdensity = np.loadtxt('PhaseSpaceDensity.txt',delimiter=',')

scanSteps = GlobalParams['ScanSteps']

psresolution = psdensity.shape[1]

nimage = int(sinograms.shape[0]/scanSteps)

sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,scanSteps,-1,1));

psdensity = psdensity[:,np.newaxis,:].reshape((nimage,psresolution,psresolution))

fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(sinograms[i], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(psdensity[i,:,:], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    
    
ntest = GlobalParams['TestSamples'];

psScale = 1000;

xtest = sinograms[:ntest,:,:,:];
print(xtest.shape)

ytest = psdensity[:ntest,:,:]/psScale;
ytest = ytest.reshape((ntest,-1))

xtrain = sinograms[ntest:,:,:,:];
print(xtrain.shape)

ytrain = psdensity[ntest:,:]/psScale;
ytrain = ytrain.reshape((nimage-ntest,-1))


linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=xtest.shape[1:]),
    layers.Flatten(),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(ytest.shape[1],activation='linear')
])


test_set = 1 + np.arange(6);

mlpredict = linear_model.predict(xtest[test_set])

fig = plt.figure(figsize=(16,5))
for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest[test_set[0]+i,:].reshape(psresolution,psresolution), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(psresolution,psresolution), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_absolute_error')


history = linear_model.fit(
    xtrain, ytrain, 
    epochs=30,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
    
    
test_set = 1 + np.arange(6);

mlpredict = linear_model.predict(xtest[test_set])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest[test_set[0]+i,:].reshape(psresolution,psresolution), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(psresolution,psresolution), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_squared_error')


history = linear_model.fit(
    xtrain, ytrain, 
    epochs=20,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


test_set = 1 + np.arange(6);

mlpredict = linear_model.predict(xtest[test_set])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest[test_set[0]+i,:].reshape(psresolution,psresolution), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredict[i,:].reshape(psresolution,psresolution), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
fname = 'TomographyFitResults' + tag + '.png'
    
plt.savefig(fname)
plt.show()
