function psx = MakePhaseSpace(params)

%Defines a new function MakePhaseSpace ready for creation.

gammax = (1 + params.alphax^2)/params.betax;

%Defining gamma as a function of beta and alpha.

covmatx = [ params.betax   -params.alphax;...
           -params.alphax          gammax ]*params.epsx;
       
%Covariance Matrix defined in terms of Twiss parameters.

       
       
% sigx    = sqrt(covmatx(1,1));

%Uncertainty of position is the square root of parameter (Beta Epsilon)

% sigpx   = sqrt(covmatx(2,2));

%Uncertainty of momentum is the square root of parameter (Gamma Epsilon)

% psrange = 0.0018; %3*max([sigx sigpx]);

xvals   = (-1:2/(params.psresn-1):1) * params.psrange;
pxvals  = (-1:2/(params.psresn-1):1) * params.psrange;

%Both parameters create a numberline between -1 and 1 with evenly 
%spaced increments of 2/(params.psresn-1).

[psxv, pspxv] = meshgrid(xvals, pxvals);

%Now creating a meshgrid of the previous parameters.

%The mesh grid function takes the numerical inputs fron xvals and pxvals
%and outputs two arrays, psxv and pspxv (Phase Space X-Values and Phase
%Space PX-Values).

%Each point from one input data set is repeated to match the number
%of data points in the second data set so lets say if xvals contains
%3 data points and pxvals contains 5, then a total of 15 data points 
%would be created as each of the first 3 data points in repeatedly mapped
%to all of the 5 secondary data points.


psv     = [psxv(:)' ; pspxv(:)'];

%Here a single array is created from the previous 2 arrays. The semi-colon
%seperates psxv and pspxv into seperate rows while the apostrophe denotes
%the transpose of each array. The colon denotes that all data is used from
%each array.

psx     = exp(-diag(psv'/covmatx*psv)/2);

%This line takes the transpose of psv, and multiplies by the inverse of the
%ovariant matrix (divide by the covariant matrix) which is then multiplied
%by psv (phase space vector). The negative trace of this value is then taken
%which is divided by two before being exponentiated.

%The trace here is taken as like elements of x and px for instance 1, 5 or
%n only impact the data as these are for the same "pixel cell". The trace
%of the matrix contains only the elements with the same row and colum
%numbers and so these elements are the only ones that correspond to te same
%cell and thus the only meaningful data.

%This now outputs all the data in a long list ready to be re-shaped next.

psx     = reshape(psx, [params.psresn, params.psresn]);

%Reshape psx to an image of params.psresn x params.psresn which in this
%case is defined as 48 x 48 elsewhere.

% imagesc(psx);
% set(gca, 'YDir', 'normal');
