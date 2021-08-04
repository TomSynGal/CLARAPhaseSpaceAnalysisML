
load TrainingImages

nvalid = 1500;

%Make a validation set for the neural network of length 1500.

XValidation = XTrain(:,:,:,1:nvalid);
YValidation = YTrain(1:nvalid,:);

%Slice the X and Y datasets to the first 1500 units to create two
%validation data sets that the neural network may use to self evaluate
%later.

XTrain = XTrain(:,:,:,nvalid+1:end);
YTrain = YTrain(nvalid+1:end,:);

%The remaining data is then used to train the neural network.

nproj  = size(XTrain,1);

%Number of projections to input to the network.

psresn = size(XTrain,2);

%Resolution Scale?

%Neural network creation.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

layers = [
    imageInputLayer([nproj psresn 1])
    
    %Input layer, Image data in to this layer.
    

    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    %A series of convolutional, batch normalised and relu layers in a
    %repeating pattern. A good archetecture as the batch normalisation
    %helps to speed up the training process.
    
    dropoutLayer(0.2)
    fullyConnectedLayer(3)
    regressionLayer];

%3 neuron output layer to output the 3 desired parameters the network is
%designed to regress to.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YTrain(:,3)      =  YTrain(:,3)*10;
YValidation(:,3) =  YValidation(:,3)*10;

miniBatchSize  = 100;

%Batch size for the batch normalisation layer.

validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('rmsprop', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',0.2e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

%20 epochs for a concice runtime.
%Low learning rate (further optimisation and potential?)
%Data shuffle active every subsequent epoch.
%Recall the validation data from the top of the script to train the network
%showing that this is a supervised learning type neural network.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

net = trainNetwork(XTrain,YTrain,layers,options);

net.Layers

save NeuralNetwork.mat net

%Save the network to disk.

return