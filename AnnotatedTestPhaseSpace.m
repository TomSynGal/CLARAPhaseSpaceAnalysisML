
global covmatx
global xvals
global pxvals
global psxv
global pspxv
global psv

psparams.epsx    = 1.0e-6;

psparams.betax   = 1.5;
psparams.alphax  =-0.2;

psparams.psresn  = 48;

psparams.psrange = 3*sqrt(psparams.betax * psparams.epsx);

phasespace       = MakePhaseSpace(psparams);

imagesc([min(xvals) max(xvals)], [min(pxvals) max(pxvals)], phasespace);
set(gca, 'YDir', 'normal');
