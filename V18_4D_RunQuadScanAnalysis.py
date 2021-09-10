import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from V18_4D_Parameters import Params
from V18_4D_MakeSinograms import MakeSinograms4D, MultipleHeatMapPlot

###############################################################################

generateSinograms = False

if generateSinograms:

    sinogramsX, opticsParamsX, SinogramsY, opticsParamsY = MakeSinograms4D(Params)

else:
        
    sinogramsX    = np.loadtxt('SinogramsX' +Params['Tag']+ '.txt',delimiter=',')
    opticsParamsX = np.loadtxt('OpticsParametersX' +Params['Tag']+ '.txt',delimiter=',')
    sinogramsY    = np.loadtxt('SinogramsY' +Params['Tag']+ '.txt',delimiter=',')
    opticsParamsY = np.loadtxt('OpticsParametersY' +Params['Tag']+ '.txt',delimiter=',')
    

#X Axis

nimage = opticsParamsX.shape[0];

sinogramsX = sinogramsX[:,np.newaxis,:,np.newaxis].reshape((nimage,Params['ScanSteps'],-1,1));

figureSize = (16, 8);

tag = Params['Tag']

fnameX = 'QuadScanSinogramsX' + tag + '.png'

figure = MultipleHeatMapPlot(sinogramsX, figureSize, fnameX)


opticsScaleX = np.diag([1/Params['emitX'], 1/Params['betaX'], 1/Params['alphaX']])
#print(opticsScaleX)

n_test = Params['TestSamples']

x_test = sinogramsX[:n_test,:,:,:];
#print(x_test.shape)

x_test_2 = np.matmul(opticsParamsX[:n_test,:],opticsScaleX);

x_train = sinogramsX[n_test:,:,:,:];
#print(x_train.shape)

x_train_2 = np.matmul(opticsParamsX[n_test:,:],opticsScaleX);


linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=x_test.shape[1:]),
    layers.Flatten(),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(3,activation='linear')
])
    
linear_model.compile(
    optimizer = tf.optimizers.Adam(learning_rate=0.5e-3),
    metrics = ['accuracy'],
    loss = 'mean_absolute_error')

history = linear_model.fit(
    x_train, x_train_2, 
    epochs = 50,
    batch_size = 50,
    # suppress logging
    verbose = 1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

print(history.history.keys())
plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy, X-PX Phase Space')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_accuracy_X' + tag + '.png')
plt.show()
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss, X-PX Phase Space')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_loss_X' + tag + '.png')
plt.show()
linear_model.save('ML_Model_X' + tag)

mlpredictX = linear_model.predict(x_test)
fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(14,8))

ax[0,0].plot(x_test_2[:,0],mlpredictX[:,0],'.');
ax[0,0].set_xlabel('True emittance');
ax[0,0].set_ylabel('Fitted emittance');

ax[0,1].plot(x_test_2[:,1],mlpredictX[:,1],'.');
ax[0,1].set_xlabel('True beta');
ax[0,1].set_ylabel('Fitted beta');

ax[0,2].plot(x_test_2[:,2],mlpredictX[:,2],'.');
ax[0,2].set_xlabel('True alpha');
ax[0,2].set_ylabel('Fitted alpha');

ratioEmittanceX = np.divide(x_test_2[:,0],mlpredictX[:,0]);
ratioEmittanceX = np.extract((ratioEmittanceX>0.2) & (ratioEmittanceX<2.0), ratioEmittanceX);
ax[1,0].hist(ratioEmittanceX,bins=np.linspace(0.5,1.5,num=20));
ax[1,0].set_xlabel('True/fitted emittance');
ax[1,0].set_ylabel('Number of cases');

ratioBetaX = np.divide(x_test_2[:,1],mlpredictX[:,1]);
ratioBetaX = np.extract((ratioBetaX>0.2) & (ratioBetaX<2.0), ratioBetaX);
ax[1,1].hist(ratioBetaX,bins=np.linspace(0.5,1.5,num=20));
ax[1,1].set_xlabel('True/fitted beta');
ax[1,1].set_ylabel('Number of cases');

ratioAlphaX = np.divide(x_test_2[:,2],mlpredictX[:,2]);
ratioAlphaX = np.extract((ratioAlphaX>0.2) & (ratioAlphaX<2.0), ratioAlphaX);
ax[1,2].hist(ratioAlphaX,bins=np.linspace(0.5,1.5,num=20));
ax[1,2].set_xlabel('True/fitted alpha');
ax[1,2].set_ylabel('Number of cases');

plt.show()

fname = 'QuadScanFitResultsX' + tag + '.png'

plt.savefig(fname)

print('for the x-px phase space region...')
print(f'Mean of true/fitted emittance = {np.mean(ratioEmittanceX):.3f}')
print(f'Standard deviation of true/fitted emittance = {100*np.std(ratioEmittanceX):.2f}%')
print()
print(f'Mean of true/fitted beta = {np.mean(ratioBetaX):.3f}')
print(f'Standard deviation of true/fitted beta = {100*np.std(ratioBetaX):.2f}%')
print()
print(f'Mean of true/fitted alpha = {np.mean(ratioAlphaX):.3f}')
print(f'Standard deviation of true/fitted alpha = {100*np.std(ratioAlphaX):.2f}%')


#Y Axis

sinogramsY = sinogramsY[:,np.newaxis,:,np.newaxis].reshape((nimage,Params['ScanSteps'],-1,1));

fnameY = 'QuadScanSinogramsY' + tag + '.png'

figure = MultipleHeatMapPlot(sinogramsY, figureSize, fnameY)


opticsScaleY = np.diag([1/Params['emitY'], 1/Params['betaY'], 1/Params['alphaY']])
#print(opticsScaleY)

y_test = sinogramsY[:n_test,:,:,:];
#print(y_test.shape)

y_test_2 = np.matmul(opticsParamsY[:n_test,:],opticsScaleY);

y_train = sinogramsY[n_test:,:,:,:];
#print(y_train.shape)

y_train_2 = np.matmul(opticsParamsY[n_test:,:],opticsScaleY);


linear_model = tf.keras.Sequential([
    layers.BatchNormalization(input_shape=y_test.shape[1:]),
    layers.Flatten(),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(200),
    layers.Dense(3,activation='linear')
])
    
linear_model.compile(
    optimizer = tf.optimizers.Adam(learning_rate=0.5e-3),
    metrics = ['accuracy'],
    loss = 'mean_absolute_error')

history = linear_model.fit(
    y_train, y_train_2, 
    epochs = 50,
    batch_size = 50,
    # suppress logging
    verbose = 1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

print(history.history.keys())
plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy, Y-PY Phase Space')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_accuracy_Y' + tag + '.png')
plt.show()
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss, Y-PY Phase Space')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_loss_Y' + tag + '.png')
plt.show()
linear_model.save('ML_Model_Y' + tag)

mlpredictY = linear_model.predict(y_test)
fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(14,8))

ax[0,0].plot(y_test_2[:,0],mlpredictY[:,0],'.');
ax[0,0].set_xlabel('True emittance');
ax[0,0].set_ylabel('Fitted emittance');

ax[0,1].plot(y_test_2[:,1],mlpredictY[:,1],'.');
ax[0,1].set_xlabel('True beta');
ax[0,1].set_ylabel('Fitted beta');

ax[0,2].plot(y_test_2[:,2],mlpredictY[:,2],'.');
ax[0,2].set_xlabel('True alpha');
ax[0,2].set_ylabel('Fitted alpha');

ratioEmittanceY = np.divide(y_test_2[:,0],mlpredictY[:,0]);
ratioEmittanceY = np.extract((ratioEmittanceY>0.2) & (ratioEmittanceY<2.0), ratioEmittanceY);
ax[1,0].hist(ratioEmittanceY,bins=np.linspace(0.5,1.5,num=20));
ax[1,0].set_xlabel('True/fitted emittance');
ax[1,0].set_ylabel('Number of cases');

ratioBetaY = np.divide(y_test_2[:,1],mlpredictY[:,1]);
ratioBetaY = np.extract((ratioBetaY>0.2) & (ratioBetaY<2.0), ratioBetaY);
ax[1,1].hist(ratioBetaY,bins=np.linspace(0.5,1.5,num=20));
ax[1,1].set_xlabel('True/fitted beta');
ax[1,1].set_ylabel('Number of cases');

ratioAlphaY = np.divide(y_test_2[:,2],mlpredictY[:,2]);
ratioAlphaY = np.extract((ratioAlphaY>0.2) & (ratioAlphaY<2.0), ratioAlphaY);
ax[1,2].hist(ratioAlphaY,bins=np.linspace(0.5,1.5,num=20));
ax[1,2].set_xlabel('True/fitted alpha');
ax[1,2].set_ylabel('Number of cases');

plt.show()

fname = 'QuadScanFitResultsY' + tag + '.png'

plt.savefig(fname)

print('for the y-py phase space region...')
print(f'Mean of true/fitted emittance = {np.mean(ratioEmittanceY):.3f}')
print(f'Standard deviation of true/fitted emittance = {100*np.std(ratioEmittanceY):.2f}%')
print()
print(f'Mean of true/fitted beta = {np.mean(ratioBetaY):.3f}')
print(f'Standard deviation of true/fitted beta = {100*np.std(ratioBetaY):.2f}%')
print()
print(f'Mean of true/fitted alpha = {np.mean(ratioAlphaY):.3f}')
print(f'Standard deviation of true/fitted alpha = {100*np.std(ratioAlphaY):.2f}%')