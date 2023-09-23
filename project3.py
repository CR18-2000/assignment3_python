
import numpy as np 
import numpy.linalg as la 
import matplotlib.pyplot as plt 
from time import time


def jacobi_step_1d(uh, fh, omega):
    n = len(uh) - 1
    n2 = n**2
    v = uh.copy()
    h = 1/n
    smax = 0
    for i in range(1,n):
        v[i] = (fh[i] + n2*(uh[i-1] + uh[i+1])) / (2*n2 + (2-i*h))
        v[i] = (1-omega)*uh[i] + omega*v[i]       # this is the weighted Jacobi part
        smax = max(smax, np.abs(v[i] - uh[i]))
    uh[:] = v[:]
    return smax


#3
"""
def two_grid(uh, fh, omega):
    v = uh.copy()
    n = len(uh)-1
    h = 1/n
    n2 = n**2

    if n == 2:
        uh[1] = fh[1]/(2*n2+(2-h))
        return 0

    jacobi_step_1d(uh, fh, omega)
    
    rh = np.zeros_like(uh)
    for i in range(1, n):
        rh[i] = fh[i] - (uh[i] * (2*n2 + (2 - i*h)) - n2*(uh[i-1] + uh[i+1]))
    rH = np.zeros(n//2+1)
    for i in range(1, len(rh)//2):
        rH[i] = (rh[2*i-1] + 2*rh[2*i] + rh[2*i+1])/4
        #if i%2 == 0:
        #    rH.append(rh[i])
    eH = np.zeros_like(rH)
    for _ in range(0,10):
        jacobi_step_1d(eH, rH, omega)
    eh = np.zeros_like(rh)
    for i in range(0,len(rH)-1):
        '''if i%2 == 0:
            eh[i] = eH[i//2]
        else:
            eh[i] = 1/2*(eH[i//2] + eH[i//2+1])'''
        eh[2*i] = eH[i]
        eh[2*i+1] = (eH[i] + eH[i+1])/2
    #v = uh.copy()
    uh[:] = uh[:] + eh[:]
    jacobi_step_1d(uh, fh, omega)
    smax = np.linalg.norm(uh-v, np.inf)
    return smax
"""


def w_cycle_step_1d(uh, fh, omega, alpha1, alpha2):
    n = len(uh)-1
    h = 1/n
    n2 = n**2
    if n == 2:     
        uh[1] = fh[1]/(2*n2 + (2-h))
        return 0
    else:
        for _ in range(0, alpha1):
            jacobi_step_1d(uh, fh, omega)
        rh = np.zeros_like(uh)
        for i in range(1, n):
            rh[i] = fh[i] - (uh[i] * (2*n2 + (2 - i*h)) - n2*(uh[i-1] + uh[i+1]))
        f2h = np.zeros(n//2+1)
        for i in range(1, len(rh)//2):
            f2h[i] = (rh[2*i-1] + 2*rh[2*i] + rh[2*i+1])/4
        u2h = np.zeros_like(f2h)
        w_cycle_step_1d(u2h, f2h, omega, alpha1, alpha2)
        w_cycle_step_1d(u2h, f2h, omega, alpha1, alpha2)
        l = np.zeros_like(uh)
        for i in range(0,len(u2h)-1):
            l[2*i] = u2h[i]
            l[2*i+1] = (u2h[i] + u2h[i+1])/2
            #uh[2*i] = uh[2*i] + u2h[i]
            #uh[2*i+1] = uh[2*i+1] + (u2h[i] + u2h[i+1])/2
        #l[n] = u2h[n//2]
        uh[:] = uh[:] + l[:]
        for _ in range(0, alpha2-1):
            jacobi_step_1d(uh, fh, omega)
        smax = jacobi_step_1d(uh, fh, omega)
    return smax


def full_mg_1d(uh, fh, omega, alpha1, alpha2, nu):
    n = len(uh)-1
    h = 1/n
    n2 = n**2
    if n == 2: 
        uh[1] = fh[1]/(2*n2 + (2-h))
        return 0
    else:
        f2h = np.zeros(n//2+1)
        for i in range(0, n//2):
            f2h[i] = (fh[2*i-1] + 2*fh[2*i] + fh[2*i+1])/4
        u2h = np.zeros_like(f2h)
        full_mg_1d(u2h, f2h, omega, alpha1, alpha2, nu)
        for i in range(0,len(u2h)-1):
            uh[2*i] = u2h[i]
            uh[2*i+1] = (u2h[i] + u2h[i+1])/2
        for _ in range(0, nu-1):
            w_cycle_step_1d(uh, fh, omega, alpha1, alpha2)
        smax = w_cycle_step_1d(uh, fh, omega, alpha1, alpha2)
    return smax


#2d
def jacobi_step_2d(uh, fh, omega):
    n = len(uh) - 1
    v = uh.copy()
    h = 1/n
    h2 = h**2
    smax = 0
    for i in range(1,n):
        for j in range(1,n):
            npos = np.linalg.norm(np.array([i,j]), 2)
            v[i,j] = (h2*fh[i, j] + (uh[i-1, j] + uh[i+1, j] + uh[i, j-1] + uh[i, j+1])) / (4 + h2*(2-h*npos))
            v[i,j] = (1-omega)*uh[i,j] + omega*v[i,j]       # this is the weighted Jacobi part
            smax = max(smax, np.abs(uh[i,j] - v[i,j]))
    uh[:,:] = v
    #uh[:] = v[:]
    return smax


def prolungation(v): #da 2h a h I_(2h)^h
    n = len(v)-1
    u = np.zeros((2*n+1, 2*n+1))
    for i in range(0, n+1):
        for j in range(0, n+1):
            u[2*i,2*j] = v[i,j]
    for i in range(0, n):
        for j in range(0, n+1):
            u[2*i+1, 2*j] = (v[i,j] + v[i+1, j])/2
    for i in range(0, n+1):
        for j in range(0, n):
            u[2*i, 2*j+1] = (v[i,j] + v[i, j+1])/2
    for i in range(0, n):
        for j in range(0, n):
            u[2*i+1, 2*j+1] = (v[i, j] + v[i+1, j] + v[i, j+1] + v[i+1, j+1])/4
    return u


def restriction(v): #I_h^(2h)
    n = len(v)-1
    u = np.zeros((n//2+1, n//2+1))
    for i in range(1, n//2):
        for j in range(1, n//2):
            u[i, j] = (v[2*i-1,2*j-1] + v[2*i-1,2*j+1] + v[2*i+1,2*j-1] + v[2*i+1,2*j+1] + 2*(v[2*i,2*j-1] + v[2*i,2*j+1] + v[2*i-1,2*j]+v[2*i+1,2*j]) + 4*v[2*i,2*j])/16
    return u

"""
def restriction_inj(v):
    n = len(v)-1
    u = np.zeros((n//2+1, n//2+1))
    for i in range(0, n+1, 2):
        for j in range(0, n+1, 2):
            u[i//2,j//2] = v[i,j]
    return u
"""

"""
def two_grid_2d(uh, fh, omega):
    v = uh.copy()
    n = len(uh)-1
    h = 1/n
    n2 = n**2
    h2 = h**2

    if n == 2:
        npos = np.linalg.norm(np.array([1,1]), 2)
        uh[1,1] = fh[1,1]/(4/h2 + (2-h*npos))
        return 0

    jacobi_step_2d(uh, fh, omega)
    
    rh = np.zeros_like(uh)
    for i in range(1, n):
        for j in range(1,n):
            npos = np.linalg.norm(np.array([i,j]), 2)
            rh[i, j] = fh[i, j] - (uh[i, j]*(4 + h2*(2-h*npos)) - uh[i-1, j] - uh[i+1, j] - uh[i, j-1] - uh[i, j+1])/h2
    rH = np.zeros(n//2+1)
    rH = restriction(rh)
    eH = np.zeros_like(rH)
    for _ in range(0,10):
        jacobi_step_2d(eH, rH, omega)
    eh = np.zeros_like(rh)
    eh = prolungation(eH)
    #v = uh.copy()
    uh[:,:] = uh[:,:] + eh[:,:]
    jacobi_step_2d(uh, fh, omega)
    smax = 0
    for i in range(0,n):
        smax = max(smax, np.linalg.norm(uh[i]-v[i], np.inf))
    return smax
"""


def w_cycle_step_2d(uh, fh, omega, alpha1, alpha2):
    n = len(uh)-1
    #n2 = n**2
    h = 1/n
    h2 = h**2
    if n == 2:
        npos = np.linalg.norm(np.array([1,1]), 2)
        uh[1,1] = fh[1,1]/(4/h2 + (2-h*npos))
        return 0
    else:
        for _ in range(0, alpha1):
            jacobi_step_2d(uh, fh, omega)
        rh = np.zeros_like(uh)
        for i in range(1, n):
            for j in range(1,n):
                npos = np.linalg.norm(np.array([i,j]), 2)
                rh[i, j] = fh[i, j] - (uh[i, j]*(4 + h2*(2-h*npos)) - uh[i-1, j] - uh[i+1, j] - uh[i, j-1] - uh[i, j+1])/h2
        f2h = restriction(rh)
        u2h = np.zeros_like(f2h)
        w_cycle_step_2d(u2h, f2h, omega, alpha1, alpha2)
        w_cycle_step_2d(u2h, f2h, omega, alpha1, alpha2)
        l = prolungation(u2h)
        uh[:,:] = uh[:,:] + l[:,:]
        for _ in range(0, alpha2-1):
            jacobi_step_2d(uh, fh, omega)
        smax = jacobi_step_2d(uh, fh, omega)
    return smax


def full_mg_2d(uh, fh, omega, alpha1, alpha2, nu):
    n = len(uh)-1
    h = 1/n
    h2 = h**2
    if n == 2:
        npos = np.linalg.norm(np.array([1,1]), 2)
        uh[1,1] = fh[1,1]/(4/h2 + (2-h*npos))
        return 0
    else:
        f2h = restriction(fh)
        u2h = np.zeros_like(f2h)
        full_mg_2d(u2h, f2h, omega, alpha1, alpha2, nu)
        uh[:,:] = prolungation(u2h)
        for _ in range(0, nu-1):
            w_cycle_step_2d(uh, fh, omega, alpha1, alpha2)
        smax = w_cycle_step_2d(uh, fh, omega, alpha1, alpha2)
    return smax