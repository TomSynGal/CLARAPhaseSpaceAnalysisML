
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Now ready to define the three parameters epsilon, alpha and beta as
%known absolute values that a neural network may deduce on its own later.

psparams.epsx    = 1.0; %0.250e-6;

%Value of epsilon.

psparams.betax   = 1.5;

%Value of the Twiss parameter beta.

psparams.alphax  =-0.2;

%Value of the Twiss parameter alpha.

psparams.psresn  = 48;

%Image resolution also added to psparams

psparams.psrange = 3*sqrt(psparams.betax * psparams.epsx); % 0.0018;

%3x root(beta*epsilon) which corresponds to 3* <x^2>, slide 3 on
%power point.

psparams.nproj   = 24;

%Number of projections (adjustments of the quadrupole magnets to generate
%a different image).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntrain    = 6000;

%Sample amount.

psparams1 = psparams;

XTrain = zeros(psparams.nproj, psparams.psresn, 1, ntrain);
YTrain = diag([psparams.epsx psparams.betax psparams.alphax])*(0.2 + 1.6*rand(3,ntrain));

%Take only the elements on the diagonal.

YTrain = YTrain';

%Make this matix equal to its transverse.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:ntrain
   
    psparams1.epsx   = YTrain(n,1);
    psparams1.betax  = YTrain(n,2);
    psparams1.alphax = YTrain(n,3);
    
    XTrain(:,:,1,n) = MakeSinogram(psparams1);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save TrainingImages.mat XTrain YTrain

%Save to disk.

fileTrainingX = fopen('TrainingDataX.txt','w');
fileTrainingY = fopen('TrainingDataY.txt','w');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:ntrain
    
    for row = 1:psparams.nproj
        
        for col = 1:psparams.psresn-1
            
            fprintf(fileTrainingX,'%1.0f,',round(XTrain(row,col,1,n)*1000));
            
        end
        
        fprintf(fileTrainingX,'%1.0f\n',round(XTrain(row,psparams.psresn,1,n)*1000));
        
    end
    
    fprintf(fileTrainingY,'%1.5f,%1.5f,%1.5f\n',YTrain(n,:));
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fclose(fileTrainingX);
fclose(fileTrainingY);

return