
"""
Created on Wed Nov 25 10:25:58 2020

@author: ivo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time as tim
import numpy.ma as ma 
import math

time = 4* 250. * 1e3 # s 
del_t = 2500. # s
n_t = int(time/del_t)

L= 2500. * 1e3 # m
del_x = 25. * 1e3 # m
n_x = int(L / del_x)

u_0 = 10. # m/s
J = 100 

C_0 = 1. # arbitrary

C_grid = np.zeros((int(time/del_t)+1, J))
C_grid[0,45:56] = C_0



########################### FT to k space ##################################
alpha = np.zeros(int(J/2))
beta = np.zeros(int(J/2))

# NOTE: our found alpha and beta are different from the AarnoutScript

for k in range(1, int(n_x/2)):
    alpha[k] = 0.0
    beta[k] = 0.0
    for x in range(n_x):
       alpha[k] = alpha[k] +  (2.0/n_x) * (C_grid[0,x] * math.cos(-2*np.pi*k* x/n_x))
       beta[k] = beta[k] +    (2.0/n_x) * (C_grid[0,x] * math.sin(-2*np.pi*k*x/n_x))

alpha[0] = np.mean(C_grid[0,:])

amps = np.zeros(int(J/2))
amps = np.sqrt(alpha**2 + beta**2)

# plotje van het spectrum 
 
# plt.figure()
# plt.title("Fourier spectrum")
# plt.plot(np.arange(0, int(n_x/2), 1) , amps,linewidth=1.0, color='green')
# plt.xlabel('k')
# plt.ylabel('amplitude')
# plt.show()


####################### time evolution ######################################
"Aarnout:  integrate the spectral equations in time"

Ck_grid = np.zeros((n_t, int(n_x/2)), dtype = np.complex)
Ck_grid[0,:] = alpha + 1j*beta 

### backwards euler 
# for t_i, t in enumerate(np.arange(0, time, del_t)):
#     for k in np.arange(1, int(J/2), 1):
        
#         dCkdt = (Ck_grid[t_i, k] - Ck_grid[t_i -1, k ] ) / del_t # Backwards Euler
#         Ck_grid[t_i, k]  = ( (-L) / (2 *np.pi * 1j * u_0 * k)) *  dCkdt
        
#         stop = 1
#     Ck_grid[t_i, 0] = Ck_grid[0,0]
    


### Matsuno (predictor corrector)

Cstar = np.zeros((n_t, int(n_x/2)), dtype = np.complex)

for t_i, t in enumerate(np.arange(0, time, del_t)):
    for k in np.arange(1, int(J/2), 1):
        
        if t != (time -del_t):
            Cstar[t_i+1, k] = (-2 * np.pi * 1j * u_0 * k / L ) * Ck_grid[t_i, k] * del_t + Ck_grid[t_i, k]
            Ck_grid[t_i+1, k] = ((-2 * np.pi * 1j * u_0 * k / L ) * Cstar[t_i+1, k]) * del_t - Ck_grid[t_i, k] 
        else:
            Cstar[0, k] = (-2 * np.pi * 1j * u_0 * k / L ) * Ck_grid[t_i, k] * del_t + Ck_grid[t_i, k]
            Ck_grid[0, k] = ((-2 * np.pi * 1j * u_0 * k / L ) * Cstar[0, k]) * del_t - Ck_grid[t_i, k] 
        

### Runge Kutta




############################ FT to real space ###############################
# introduce actual distances
Lx = np.zeros((n_x))
for x in range(n_x):
 Lx[x] = x * del_x

#%%

# Ck_grid = ma.masked_invalid(Ck_grid)
Cout = np.zeros(n_x, dtype = np.complex)

for x in range(n_x):
    C_grid[-1,x] = 0.0
    for k in np.arange(1, int(J/2), 1):
        C_grid[-1, x] =  C_grid[-1,x] + np.real( Ck_grid[-1,k] * (np.cos(2*np.pi*k*Lx[x]/L)) + 1j * np.sin(2*np.pi*k*Lx[x]/L))
        Cout[x] = Cout[x] + np.real( Ck_grid[-1,k] * (np.cos(2*np.pi*k*Lx[x]/L)) + 1j * np.sin(2*np.pi*k*Lx[x]/L))
        
C_grid[-1, :] = Cout

#%%
plt.figure()
# plt.axis([0,2500,0,2])
plt.title("AFTER")
plt.plot(Lx/1000,Cout,linewidth=1.0, color='red')
plt.show()


##################### animation #############################################



# fig = plt.figure()
# ax = plt.axes(xlim=(L), ylim=(-1,1.5))
# x = np.arange(0,L,del_x)
    
# #initial line, which are the initial values of eta
# line, = ax.plot(x, C_grid[0,:])

# def init():  # only required for bl  itting to give a clean slate.
#     line.set_ydata([] * len(x)) #length of the line we want to plot, [] *len(x) will make an empty array[], of lenght x
#     return line,

# def animate(n):
#     line.set_ydata(C_grid[n,:])  # update the data.
#     ax.set_title('%03d'%(n))
#     ax.plot(x, C_grid[0,:], color = 'red', ls = ':')
#     if n == 101 or n == 201 or n == 301:
#         tim.sleep(1)
#     return line,

# anim = FuncAnimation(fig, animate, init_func=init,interval=20, blit = False)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# anim.save(f'Euler_dt{del_t}_dx{del_x}.mp4', writer=writer)




