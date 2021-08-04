
clear

load NeuralNetwork
load TrainingImages

%Data load from other scripts.

nvalid = 1500;

%Define the length of the validation data set.

XValidation = XTrain(:,:,:,1:nvalid);
YValidation = YTrain(1:nvalid,:);

%Cut the full data from the start to the length of the validation set to
%generate two sets of validation data of length 1500.

% XTrain = XTrain(:,:,:,nvalid+1:end);
% YTrain = YTrain(nvalid+1:end,:);

YPredicted      = predict(net,XValidation);
YPredicted(:,3) = YPredicted(:,3)/10;

predictionError = YValidation - YPredicted;

%Making predictions from the neural network and calculating the associated
%error of these measurements.

close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Start the generation of figures.

figure
subplot(3,2,1)
plot(YValidation(:,1),YPredicted(:,1),'.k')
xlabel('Actual emittance')
ylabel('Predicted emittance')
subplot(3,2,2)
relerror = predictionError(:,1)./YValidation(:,1);
histogram(relerror,-0.1:0.005:0.1)
xlabel('Relative error')
ylabel('Counts')
title(['StDev = ' num2str(std(relerror)*100,2) '%'])

%Two sub plots in this figure showing the predicted emittance as a function
%of the actual emittance with a sub plot of the error of the predicted
%emittence also included to show the deviation from the true value.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,2,3)
plot(YValidation(:,2),YPredicted(:,2),'.k')
xlabel('Actual beta')
ylabel('Predicted beta')
subplot(3,2,4)
relerror = predictionError(:,2)./YValidation(:,2);
histogram(relerror,-0.1:0.005:0.1)
xlabel('Relative error')
ylabel('Counts')
title(['StDev = ' num2str(std(relerror)*100,2) '%'])

%Two sub plots in this figure showing the predicted beta as a function
%of the actual beta with a sub plot of the error of the predicted
%beta also included to show the deviation from the true value.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,2,5)
plot(YValidation(:,3),YPredicted(:,3),'.k')
xlabel('Actual alpha')
ylabel('Predicted alpha')
subplot(3,2,6)
relerror = predictionError(:,3)./YValidation(:,3);
histogram(relerror,-0.1:0.005:0.1)
xlabel('Relative error')
ylabel('Counts')
title(['StDev = ' num2str(std(relerror)*100,2) '%'])

%Two sub plots in this figure showing the predicted alpha as a function
%of the actual alpha with a sub plot of the error of the predicted
%alpha also included to show the deviation from the true value.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(gcf,'PaperUnits','inches')
set(gcf,'PaperPosition',[0 0 6 6])
print('-dpng','Example-Validation.png','-r400')
