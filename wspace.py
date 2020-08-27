'''
T: 2020.06.22 16:02
A: bell_liang
'''

import numpy as np
from math import pi

'''
# This function constructs a linearly-spaced vector of angular
# frequencies that correspond to the points in an FFT spectrum.
# The second half of the vector is aliased to negative
# frequencies.
# 
# USAGE
#
# w = wspace(tv);
# w = wspace(t,nt);
#
# INPUT
#
# tv - vector of linearly-spaced time values
# t - scalar representing the periodicity of the time sequence
# nt - Number of points in time sequence 
#      (should only be provided if first argument is scalar)
#
# OUTPUT
#
# w - vector of angular frequencies
# 
# EXAMPLE
#
#   t = linspace(-10,10,2048)';   # set up time vector
#   x = exp(-t.^2);               # construct time sequence
#   w = wspace(t);                # construct w vector
#   Xhat = fft(x);                # calculate spectrum
#   plot(w,abs(Xhat))             # plot spectrum

'''

def wspace(t, nt=0):
    if nt == 0:
        nt = len(t)
        dt = t[1] - t[0]
        t = t[nt-1] - t[0] + dt
    else:
        dt = t/nt
    
    w = 2*pi*np.arange(0, nt)/t
    kv = np.where(w >= pi/dt)
    w[kv] = w[kv] - 2*pi/dt
    return w