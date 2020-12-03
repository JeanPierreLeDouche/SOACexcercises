# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:45:55 2020

@author: Gebruiker
"""

# FOURIER ANALYSIS of top-hat profile  (ex. 2 SOAC)
import numpy as N
import math as M
import matplotlib.pyplot as plt

# CONSTANTS
pi = M.pi
dx = 25000
L = 2500000
nx_len = int(L/dx)        # number of grid points
nk_len = int(nx_len/2)    # number of wave numbers


# Fourier coefficients
alfa = N.zeros((nk_len))
beta = N.zeros((nk_len))

# Concentration
C = N.zeros((nx_len))
# distance
x = N.zeros((nx_len))
for nx in range(nx_len):
 x[nx] = nx * dx
 if x[nx] >=1125000 and x[nx] <1375000: C[nx] = 1.0    # "TOP-HAT"

# PLOT CONCENTRATION AS A FUNCTION OF X
plt.axis([0,2500,0,2])
plt.title("BEFORE")
plt.plot(x/1000,C,linewidth=1.0, color='red')
plt.show()

# Compute Fourier coefficients
for nk in range(1,nk_len):
  alfa[nk] = 0.0
  beta[nk] = 0.0
  for nx in range(nx_len):
   alfa[nk] = alfa[nk] + (2.0/nx_len) * (C[nx] * M.cos(-2*pi*nk*nx/nx_len))
   beta[nk] = beta[nk] + (2.0/nx_len) * (C[nx] * M.sin(-2*pi*nk*nx/nx_len))

alfa[0] = N.mean(C,axis=0)

AMPLITUDE = N.zeros((nk_len))
for nk in range(nk_len):
 AMPLITUDE[nk] = M.pow((M.pow(alfa[nk],2) + M.pow(beta[nk],2)),0.5)

# PRINT SPECTRUM
wavenumber = N.zeros((nk_len))
print ("spectrum")
print ("    nk   alfa   beta   AMPLITUDE")
for nk in range(nk_len):
 wavenumber[nk] = nk
 print ("%5.0f,%8.3f,%8.3f,%8.3f" %  (nk,alfa[nk],beta[nk],AMPLITUDE[nk]))

# PLOT SPECTRUM: AMPLITUDE AS A FUNCTION OF THE WAVENUMBER
plt.figure()
plt.title("SPECTRUM")
plt.plot(wavenumber,AMPLITUDE,linewidth=1.0, color='red')
plt.show()



# BACK-TRANSFORMATION TO PHYSICAL SPACE
for nx in range(nx_len):
 C[nx] = 0.0
 for nk in range(nk_len):
   C[nx] =  C[nx] + (alfa[nk] * M.cos(2*pi*nk*x[nx]/L)) - (beta[nk] * M.sin(2*pi*nk*x[nx]/L))

# PLOT CONCENTRATION AS A FUNCTION OF X AFTER BACK-TRANSFORMATION (SHOULD IDENTICAL TO FIRST PLOT)
plt.figure()
plt.axis([0,2500,0,2])
plt.title("AFTER")
plt.plot(x/1000,C,linewidth=1.0, color='red')
plt.show()
