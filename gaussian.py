'''
T: 2020.06.22 16:12
A: bell_liang
'''

import math
import numpy as np

'''
# This function computes a gaussian or supergaussian pulse with
# the specified parameters:
#
#   sqrt(P0) * exp{ -[(1+iC)/2] * [(t-t0)/T]^(2m) }
#      where T = FWHM/2*sqrt(log(2))
# 
# USAGE:
# 
# u = gaussian (t);
# u = gaussian (t,t0);
# u = gaussian (t,t0,FWHM);
# u = gaussian (t,t0,FWHM,P0);
# u = gaussian (t,t0,FWHM,P0,m);
# u = gaussian (t,t0,FWHM,P0,m,C);
# 
# INPUT:
# 
# t     vector of times at which to compute u
# t0    center of pulse (default = 0)
# FWHM  full-width at half-intensity of pulse (default = 1)
# P0    peak intensity (|u|^2 at t=t0) of pulse (default = 1)
# m     Gaussian order (default = 1)
# C     chirp parameter (default = 0)
# 
# OUTPUT:
# 
# u     vector of the same size as t, representing pulse
#       amplitude
#
# NOTES:
#
# In order to produce a standard gaussian pulse of the form
# exp(-t.^2/2), you can use:  
#
# u = gaussian(t,0,2*sqrt(log(2)));
'''

def gaussian(t, t0=0, FWHM=1, P0=1, m=1, C=0):
    return math.sqrt(P0)*np.exp(-((1+1j*C)/2)*(2*(t-t0)/FWHM)**(2*m))