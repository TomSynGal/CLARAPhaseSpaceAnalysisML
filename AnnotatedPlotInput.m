
clear
close all

%Clear and close all founctions to empty the workspace ready to start the
%task with a blank workspace.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Now ready to define the three parameters epsilon, alpha and beta as
%known absolute values that a neural network may deduce on its own later.

psparams.epsx    = 1.0; %0.250e-6;

psparams.betax   = 1.5;

psparams.alphax  =-1.2;

%The set of values psparams (Phase Space Parameters) now contains 3 values.

%Data structure.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

psparams.psresn  = 48;

%Image resolution also added to psresn.

psparams.psrange = 3*sqrt(psparams.betax * psparams.epsx); % 0.0018;

%3x root(beta*epsilon) which corresponds to 3* <x^2>, slide 3 on
%power point.

psparams.nproj   = 24;

%Number of projections (adjustments of the quadrupole magnets to generate
%a different image).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[sinogram, psx] = MakeSinogram(psparams);

%Calling the MakeSinogram function.

figure(1)
imagesc(psx)

%Display image with scaled colours.

set(gca,'XTick',[])
set(gca,'YTick',[])

%current axes handle.

set(gcf,'PaperUnits','inches')
set(gcf,'PaperPosition',[0 0 3 3])

%Current figure handle.

print('-dpng','Example-PhaseSpace.png','-r400')

%Creation of the plots for the phase space images.


figure(2)
imagesc(sinogram)
set(gca,'XTick',[])
set(gca,'YTick',[])

%Current axes handle.

set(gcf,'PaperUnits','inches')
set(gcf,'PaperPosition',[0 0 3 3])

%Current figure handle.

print('-dpng','Example-Sinogram.png','-r400')

%Creation of the plots for the sinograms.

