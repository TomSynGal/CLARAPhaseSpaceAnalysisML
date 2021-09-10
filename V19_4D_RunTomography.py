import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from V19_4D_Parameters import Params
from V19_4D_MakeSinograms import MakeSinograms4D

Params['ComplexPhaseSpace'] = True

generateSinograms = True


if generateSinograms:

    sinogramsX, psdensityX, sinogramsY, psdensityY = MakeSinograms4D(Params)

else:
        
    sinogramsX = np.loadtxt('SinogramsComplexPSX' +Params['Tag']+ '.txt',delimiter=',')
    psdensityX = np.loadtxt('PhaseSpaceDensityX' +Params['Tag']+ '.txt',delimiter=',')
    sinogramsY = np.loadtxt('SinogramsComplexPSY' +Params['Tag']+ '.txt',delimiter=',')
    psdensityY = np.loadtxt('PhaseSpaceDensityY' +Params['Tag']+ '.txt',delimiter=',')

#X-PX Phase Space

scanSteps = Params['ScanSteps']

psresolutionX = psdensityX.shape[1]

nimageX = int(sinogramsX.shape[0]/scanSteps)

sinogramsX = sinogramsX[:,np.newaxis,:,np.newaxis].reshape((nimageX,scanSteps,-1,1));

psdensityX = psdensityX[:,np.newaxis,:].reshape((nimageX,psresolutionX,psresolutionX))

fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(sinogramsX[i], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(psdensityX[i,:,:], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    
    
ntest = Params['TestSamples'];

psScale = 1000;

xtest = sinogramsX[:ntest,:,:,:];
print(xtest.shape)

xtest_2 = psdensityX[:ntest,:,:]/psScale;
xtest_2 = xtest_2.reshape((ntest,-1))

xtrain = sinogramsX[ntest:,:,:,:];
print(xtrain.shape)

xtrain_2 = psdensityX[ntest:,:]/psScale;
xtrain_2 = xtrain_2.reshape((nimageX-ntest,-1))


linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=xtest.shape[1:]),
    layers.Flatten(),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(xtest_2.shape[1],activation='linear')
])


test_setX = 1 + np.arange(6);

mlpredictX = linear_model.predict(xtest[test_setX])

fig = plt.figure(figsize=(16,5))
for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(xtest_2[test_setX[0]+i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictX[i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_absolute_error')


history = linear_model.fit(
    xtrain, xtrain_2, 
    epochs=30,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
    
    
test_setX = 1 + np.arange(6);

mlpredictX = linear_model.predict(xtest[test_setX])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(xtest_2[test_setX[0]+i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictX[i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_squared_error')


history = linear_model.fit(
    xtrain, xtrain_2, 
    epochs=20,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


test_setX = 1 + np.arange(6);

mlpredictX = linear_model.predict(xtest[test_setX])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(xtest_2[test_setX[0]+i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictX[i,:].reshape(psresolutionX,psresolutionX), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
fname = 'TomographyFitResultsX' + Params['Tag'] + '.png'
    
plt.savefig(fname)
plt.show()


#Y-PY Phase Space

scanSteps = Params['ScanSteps']

psresolutionY = psdensityY.shape[1]

nimageY = int(sinogramsY.shape[0]/scanSteps)

sinogramsY = sinogramsY[:,np.newaxis,:,np.newaxis].reshape((nimageY,scanSteps,-1,1));

psdensityY = psdensityY[:,np.newaxis,:].reshape((nimageY,psresolutionY,psresolutionY))

fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(sinogramsY[i], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


fig = plt.figure(figsize=(16,5))
for i in range(24):
    sub = fig.add_subplot(3, 8, i + 1)
    sub.imshow(psdensityY[i,:,:], interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    
    
ntest = Params['TestSamples'];

psScale = 1000;

ytest = sinogramsY[:ntest,:,:,:];
print(ytest.shape)

ytest_2 = psdensityY[:ntest,:,:]/psScale;
ytest_2 = ytest_2.reshape((ntest,-1))

ytrain = sinogramsY[ntest:,:,:,:];
print(ytrain.shape)

ytrain_2 = psdensityY[ntest:,:]/psScale;
ytrain_2 = ytrain_2.reshape((nimageY-ntest,-1))


linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=ytest.shape[1:]),
    layers.Flatten(),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(300),
    layers.Dense(ytest_2.shape[1],activation='linear')
])


test_setY = 1 + np.arange(6);

mlpredictY = linear_model.predict(ytest[test_setY])

fig = plt.figure(figsize=(16,5))
for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest_2[test_setY[0]+i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictY[i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest', aspect='auto')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_absolute_error')


history = linear_model.fit(
    ytrain, ytrain_2, 
    epochs=30,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)
    
    
test_setY = 1 + np.arange(6);

mlpredictY = linear_model.predict(ytest[test_setY])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest_2[test_setY[0]+i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictY[i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])


linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1e-3),
    loss='mean_squared_error')


history = linear_model.fit(
    ytrain, ytrain_2, 
    epochs=20,
    batch_size=20,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)


test_setY = 1 + np.arange(6);

mlpredictY = linear_model.predict(ytest[test_setY])

fig = plt.figure(figsize=(16,5))

for i in range(6):
    sub = fig.add_subplot(2, 6, i + 1)
    sub.imshow(ytest_2[test_setY[0]+i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    sub = fig.add_subplot(2, 6, i + 7)
    sub.imshow(mlpredictY[i,:].reshape(psresolutionY,psresolutionY), interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    
fname = 'TomographyFitResultsY' + Params['Tag'] + '.png'
    
plt.savefig(fname)
plt.show()