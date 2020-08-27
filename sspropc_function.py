'''
2020.06.21 23:06
bell_liang
'''

import numpy as np
import scipy.fftpack as fp
import math

'''
# USAGE
#
# u1 = sspropc(u0,dt,dz,nz,alpha,betap,gamma);
# u1 = sspropc(u0,dt,dz,nz,alpha,betap,gamma,tr);
# u1 = sspropc(u0,dt,dz,nz,alpha,betap,gamma,tr,to);
# u1 = sspropc(u0,dt,dz,nz,alpha,betap,gamma,tr,to,maxiter);
# u1 = sspropc(u0,dt,dz,nz,alpha,betap,gamma,tr,to,maxiter,tol);
#
# INPUT
#
# u0        starting field amplitude (vector)
# dt        time step
# dz        propagation stepsize
# nz        number of steps to take, ie, ztotal = dz*nz
# alpha     power loss coefficient, ie, P=P0*exp(-alpha*z)
# betap     dispersion polynomial coefs, [beta_0 ... beta_m]
# gamma     nonlinearity coefficient
# tr        Raman response time (default = 0)
# to        optical cycle time = lambda0/c (default = 0)
# maxiter   max number of iterations (default = 4)
# tol       convergence tolerance (default = 1e-5)
#
# The loss coefficient alpha may optionally be specified as a
# vector of the same length as u0, in which case it is treated as
# vector that describes alpha(w) in the frequency domain.  (The
# function wspace.m can be used to construct a vector of the
# corresponding frequencies.)
#
# Similarly, the propagation constant beta(w) can be specified
# directly by replacing the polynomial argument betap with a
# vector of the same length as u0.  In this case, the argument
# betap is treated as a vector describing propagation in the
# frequency domain. 
#
# OUTPUT
#
# u1        field at the output
'''

def abs2(x):
    return x.real**2 + x.imag**2

def prodr(x, y):
    return x.real*y.real + x.imag*y.imag

def prodi(x, y):
    return x.real*y.imag - x.imag*y.real

def round(x):
    return int(x+0.5)

def ssconverged(a, b, t, nt):
    num = 0.
    denom = 0.
    for jj in range(0, nt):
        denom += b[jj].real**2 + b[jj].imag**2
        num += (b[jj].real - a[jj].real/nt)**2 + (b[jj].imag - a[jj].imag/nt)**2
    p = 1 - num/denom
    return p < t

def sspropc(u0, dt, dz, nz, alpha, beta, gamma, tr=0, to=0, maxiter=4, tol=1e-5):
    traman = tr
    toptical = to

    nalpha = len(alpha)
    nbeta = len(beta)
    nt = len(u0)
    
    halfstep = np.arange(0, nt) + .1 + 1j
    u1 = np.arange(0, nt) + .1 + 1j
    u2 = np.arange(0, nt) + .1 + 1j
    uv = np.arange(0, nt) + 1. + 1j
    
    # w = wspace(tv)
    w = np.arange(0, nt) + 1.
    for ii in range(0, int((nt-1)/2+1)):
        w[ii] = 2*math.pi*ii/(dt*nt)
    for ii in range(int((nt-1)/2+1), nt):
        w[ii] = 2*math.pi*ii/(dt*nt) - 2*math.pi/dt
    
    # compute halfstep and initialize u0 and u1
    for jj in range(0, nt):
        phase = 0
        if nbeta != nt:
            fii = 1
            wii = 1
            time = 1
            for ii in range(0, nbeta):
                phase += wii*beta[ii]/fii
                fii *= time
                time += 1
                wii *= w[jj]
        else:
            phase = beta[jj]
        
        if nalpha == nt:
            temp_alpha = alpha[jj]
        else:
            temp_alpha = alpha[0]
        
        halfstep[jj] = math.exp(-temp_alpha*dz/4)*math.cos(phase*dz/2)
        halfstep[jj] += -math.exp(-temp_alpha*dz/4)*math.sin(phase*dz/2)*1j

        u1[jj] = u0[jj]
        u2[jj] = u0[jj]
    
    ################################################
    ufft = fp.fft(u0)
    for _ in range(0, nz):
        uhalf = halfstep*ufft
        uhalf = fp.ifft(uhalf)
        inner_iter = 0
        for ii in range(0, maxiter):
            if traman == 0 and toptical == 0:
                for jj in range(0, nt):
                    phase = gamma*(u1[jj].real**2 + u1[jj].imag**2+ u2[jj].real**2 + u2[jj].imag**2)*dz/2
                    uv[jj] = (uhalf[jj].real*math.cos(phase) + uhalf[jj].imag*math.sin(phase))/nt
                    uv[jj] += (-uhalf[jj].real*math.sin(phase) + uhalf[jj].imag*math.cos(phase))/nt*1j
            else:
                # the first point jj = 0
                jj = 0
                ua = u1[nt-1]
                ub = u1[jj]
                uc = u1[jj+1]
                nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                ua = u2[nt-1]
                ub = u2[jj]
                uc = u2[jj+1]
                nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                nlp_real *= gamma*dz/2
                nlp_imag *= gamma*dz/2

                uv[jj] = (uhalf[jj].real*math.cos(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.sin(nlp_real)*math.exp(nlp_imag))/nt
                uv[jj] += (-uhalf[jj].real*math.sin(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.cos(nlp_real)*math.exp(nlp_imag))/nt*1j

                # the points jj = range(1, nt-1)
                for jj in range(1, nt-1):
                    ua = u1[jj-1]
                    ub = u1[jj]
                    uc = u1[jj+1]
                    nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                    nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                    ua = u2[jj-1]
                    ub = u2[jj]
                    uc = u2[jj+1]
                    nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                    nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                    nlp_real *= gamma*dz/2
                    nlp_imag *= gamma*dz/2

                    uv[jj] = (uhalf[jj].real*math.cos(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.sin(nlp_real)*math.exp(nlp_imag))/nt
                    uv[jj] += (-uhalf[jj].real*math.sin(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.cos(nlp_real)*math.exp(nlp_imag))/nt*1j
                
                # the endpoint jj = nt - 1
                jj = nt - 1
                ua = u1[jj-1]
                ub = u1[jj]
                uc = u1[0]
                nlp_imag = -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                nlp_real = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                ua = u2[jj-1]
                ub = u2[jj]
                uc = u2[0]
                nlp_imag += -toptical*(abs2(uc) - abs2(ua) + prodr(ub, uc) - prodr(ub, ua)) / (4*math.pi*dt)
                nlp_real += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub, uc) - prodi(ub, ua))/(4*math.pi*dt)

                nlp_real *= gamma*dz/2
                nlp_imag *= gamma*dz/2

                uv[jj] = (uhalf[jj].real*math.cos(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.sin(nlp_real)*math.exp(nlp_imag))/nt
                uv[jj] += (-uhalf[jj].real*math.sin(nlp_real)*math.exp(nlp_imag) + uhalf[jj].imag*math.cos(nlp_real)*math.exp(nlp_imag))/nt*1j
            
            uv = fp.fft(uv)
            ufft = uv*halfstep
            uv = fp.ifft(ufft)

            if ssconverged(uv, u1, tol, nt):
                u2 = uv/nt
                break
            else:
                u2 = uv/nt
                inner_iter += 1
            
        u1 = u2
    
    print("done.\n")
    return u1
