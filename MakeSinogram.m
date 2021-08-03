function [sinogram, psx] = MakeSinogram(params)

%Creation of the MakeSinogram function taking the input argument params and
%outputting arguments sinogram and psx.

psx       = MakePhaseSpace(params);

%The psx argument for the MakeSinogram function calls upon another function
%MakePhaseSpace 

% figure(1)
% imagesc(psx);
% set(gca, 'YDir', 'normal');

nproj     = params.nproj;

%Adds nproj to params.

sinogram  = zeros(nproj, params.psresn);

%Sinogram is now reprisented by a nprij x params.psresn matrix of zeros.

for n = 1:nproj
    
    %For loop of n from values of 1 to nproj (24) in steps of 1.
   
    theta   = (n-1)*180/nproj;
    
    %Defining the angle theta as n multiples of 180 degrees over the
    %amount of different projections taken, nproj (24).
    
    psxrotn = imrotate(psx, theta, 'bilinear', 'crop');
    
    %Rotate psx by the angle theta with bilinear interpolation with the
    %request that the image be cropped to the same as the original.
    
    psxproj = sum(psxrotn, 1);
    
    %Lists all projection results.
    
    sinogram(n,:) = psxproj;
    
    %Update all of sinograms with the projections.
    
end

% figure(2)
% imagesc(sinogram)
