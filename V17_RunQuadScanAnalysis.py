import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from V17_Parameters import GlobalParams
from V17_MakeSinograms import MakeSinograms, MultipleHeatMapPlot

###############################################################################

generateSinograms = True

if generateSinograms:

    sinograms, opticsParams = MakeSinograms(GlobalParams)

else:
        
    sinograms    = np.loadtxt('Sinograms.txt',delimiter=',')
    opticsParams = np.loadtxt('OpticsParameters.txt',delimiter=',')

nimage = opticsParams.shape[0];

sinograms = sinograms[:,np.newaxis,:,np.newaxis].reshape((nimage,GlobalParams['ScanSteps'],-1,1));

figureSize = (16, 8);

tag = GlobalParams['Tag']

fname = 'QuadScanSinograms' + tag + '.png'

figure = MultipleHeatMapPlot(sinograms, figureSize, fname)


opticsScale = np.diag([1/GlobalParams['emitX'], 1/GlobalParams['betaX'], 1/GlobalParams['alphaX']])
#print(opticsScale)

n_test = GlobalParams['TestSamples']

x_test = sinograms[:n_test,:,:,:];
#print(x_test.shape)

y_test = np.matmul(opticsParams[:n_test,:],opticsScale);

x_train = sinograms[n_test:,:,:,:];
#print(x_train.shape)

y_train = np.matmul(opticsParams[n_test:,:],opticsScale);


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
    x_train, y_train, 
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
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_accuracy' + tag + '.png')
plt.show()
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig('model_loss' + tag + '.png')
plt.show()
linear_model.save('ML_Model' + tag)

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
ratioEmittance = np.extract((ratioEmittance>0.2) & (ratioEmittance<2.0), ratioEmittance);
ax[1,0].hist(ratioEmittance,bins=np.linspace(0.5,1.5,num=20));
ax[1,0].set_xlabel('True/fitted emittance');
ax[1,0].set_ylabel('Number of cases');

ratioBeta = np.divide(y_test[:,1],mlpredict[:,1]);
ratioBeta = np.extract((ratioBeta>0.2) & (ratioBeta<2.0), ratioBeta);
ax[1,1].hist(ratioBeta,bins=np.linspace(0.5,1.5,num=20));
ax[1,1].set_xlabel('True/fitted beta');
ax[1,1].set_ylabel('Number of cases');

ratioAlpha = np.divide(y_test[:,2],mlpredict[:,2]);
ratioAlpha = np.extract((ratioAlpha>0.2) & (ratioAlpha<2.0), ratioAlpha);
ax[1,2].hist(ratioAlpha,bins=np.linspace(0.5,1.5,num=20));
ax[1,2].set_xlabel('True/fitted alpha');
ax[1,2].set_ylabel('Number of cases');

fname = 'QuadScanFitResults' + tag + '.png'

plt.savefig(fname)

print(f'Mean of true/fitted emittance = {np.mean(ratioEmittance):.3f}')
print(f'Standard deviation of true/fitted emittance = {100*np.std(ratioEmittance):.2f}%')
print()
print(f'Mean of true/fitted beta = {np.mean(ratioBeta):.3f}')
print(f'Standard deviation of true/fitted beta = {100*np.std(ratioBeta):.2f}%')
print()
print(f'Mean of true/fitted alpha = {np.mean(ratioAlpha):.3f}')
print(f'Standard deviation of true/fitted alpha = {100*np.std(ratioAlpha):.2f}%')
