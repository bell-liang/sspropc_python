'''
T: 2020.06.22 15:57
A: bell_liang
'''

from sspropc_function import sspropc
from wspace import wspace
from gaussian import gaussian
import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

T = 100
nt = 2**10
dt = T/nt
t = (np.arange(0, nt) - (nt-1)/2)*dt
w = wspace(T, nt)
vs = fp.fftshift(w/(2*math.pi))

z = 3*(math.pi/2)
nz = 5000
nplot = 3
n1 = round(nz/nplot + 0.5)
nz = n1*nplot
dz = z/nz

alpha = np.array([0])

beta2 = -1
beta3 = .03
betap = np.array([0, 0, beta2, beta3])
s = 0.05
tr = 0.1
N = 2

u0 = gaussian(t)

zv = (z/nplot)*np.arange(0, nplot+1)
u = np.zeros((len(t), len(zv))) + 0j + 0.
U = np.zeros((len(t), len(zv))) + 0j + 0.

u[:, 1] = u0
U[:, 1] = fp.fftshift(abs(dt*fp.fft(u[:, 1])/math.sqrt(2*math.pi))**2)

for ii in range(1, nplot):
    u[:, ii+1] = sspropc(u[:, ii], dt, dz, n1, alpha, betap, N**2, tr, 2*math.pi*s)
    U[:, ii+1] = fp.fftshift(abs(dt*fp.fft(u[:, ii+1])/math.sqrt(2*math.pi))**2)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_ylim(-25,25) 
X = zv/(math.pi/2)
Y = t
X, Y = np.meshgrid(X, Y)
Z = abs(u)**2
ax.plot_wireframe(X, Y, Z, cmap='rainbow')

fig2 = plt.figure()
ax2 = Axes3D(fig2)
X2 = zv/(math.pi/2)
Y2 = vs
Z2 = U
X2, Y2 = np.meshgrid(X2, Y2)
ax2.plot_surface(X2, Y2, Z2, cmap='rainbow')

plt.show()
