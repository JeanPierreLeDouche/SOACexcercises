

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
from os.path import join

time = 1* 250. * 1e3 # s 
del_t = 50
n_t = int(time/del_t)

L= 2500. * 1e3 # m
del_x = 25. * 1e3 # m
n_x = int(L / del_x)

u_0 = 10. # m/s
J = 100 

# introduce actual distances
Lx = np.zeros((n_x))
for x in range(n_x):
    Lx[x] = x * del_x

#### CFL criterion
CFL = u_0 * del_t/del_x
if CFL > 1:
    print('You a silly boy, use smaller del_t or bigger del_x quickly now !')

#-----------------------------------------------------------------------------
######################## choose a scheme ! ###################################
#-----------------------------------------------------------------------------

########## spectral schemes: 

# scheme = 'SBE'  # spectral method w/ backwards euler
# scheme = 'SM'  # spectral method w/ Matsuno
# scheme = 'SRK4' # spectral method w/ Runge Kutta 4 ( to be implemented)
scheme = 'SFE' # spectral method w/ forward euler 

#-----------------------------------------------------------------------------
##############################################################################
#-----------------------------------------------------------------------------

C_0 = 1. # arbitrary

C_init = np.zeros(J)
C_init[45:55] = C_0 

plt.figure()
plt.plot(Lx, C_init)
plt.title('Top-hat')
plt.xlabel('x')
plt.ylabel('C')
plt.savefig('figs/Tophat')
plt.show()

########################### FT to k space ##################################
alpha = np.zeros(int(J/2))
beta = np.zeros(int(J/2))

# NOTE: our found alpha and beta are different from the AarnoutScript
# NOTE2: this was caused by Aarnouts top hat running from j=45 to 54 and NOT 55 (which we did),
# making the finite change symmetrical around the tophat except the middle value of 1e-16
# which was all that we had left for beta

for k in range(1, int(n_x/2)):
    alpha[k] = 0.0
    beta[k] = 0.0
    for x_i in range(n_x):
       alpha[k]=  alpha[k] +  (2.0/n_x) * (C_init[x_i] * math.cos(-2*math.pi*k*x_i/n_x))
       beta[k] =  beta[k] +   (2.0/n_x) * (C_init[x_i] * math.sin(-2*math.pi*k*x_i/n_x))
       
alpha[0] = np.mean(C_init)

amps = np.zeros(int(J/2))
amps = np.sqrt(alpha**2 + beta**2)

# plotjes van het spectrum 
 
plt.figure()
plt.title("Fourier spectrum")
plt.plot(np.arange(0, int(n_x/2), 1) , amps,linewidth=1.0, color='green')
plt.xlabel('k')
plt.ylabel('amplitude')
plt.savefig('figs/FourierSpectrumBEFORE')

plt.show()

####################### time evolution ######################################
"Aarnout:  integrate the spectral equations in time"

Ck_grid = np.zeros((n_t, int(n_x/2)), dtype = np.complex)
Ck_init = np.zeros(int(n_x/2), dtype = np.complex)

Ck_init = (alpha + 1j*beta).transpose()
Ck_grid[0,:] = (alpha + 1j*beta).transpose()
Ck_grid[:,0] = alpha[0]


### backwards euler 
if scheme == 'SBE':
    subtitle = r' spectral method, backward Euler '
    for t_i, t in enumerate(np.arange(0, time, del_t)):
        for k in np.arange(1, int(J/2), 1):
            
            dCkdt = (Ck_grid[t_i, k] - Ck_grid[t_i -1, k ] ) / del_t # Backwards Euler
            Ck_grid[t_i, k]  = ( (-L) / (2 *np.pi *(0 + 1j) * u_0 * k)) *  dCkdt
            
### forward euler
if scheme == 'SFE':
    subtitle = r' spectral method, forward Euler '

    for t_i in np.arange(0, int(time/del_t) - 1, 1): 
        for k in np.arange(1, int(J/2), 1):
            Ck_grid[t_i+1, k] = Ck_grid[t_i,k] + del_t * (-2 * np.pi * 1j* u_0 * k) / L * Ck_grid[t_i,k] 
        
### Matsuno (predictor corrector)
if scheme == 'SM':
    subtitle = r' spectral method, Matsuno'
    Cstar = np.zeros((n_t, int(n_x/2)), dtype = np.complex)
    change_grid = np.zeros((n_t, int(n_x/2)), dtype = np.complex)
    
    for t_i in np.arange(0, int(time/del_t) -1 , 1):
        for k in np.arange(1, int(J/2), 1):
            Cstar[t_i + 1, k] =  ( -2 * np.pi * (0 +1j ) * u_0 * k * del_t) * Ck_grid[t_i, k]   / L + Ck_grid[t_i, k]        
            Ck_grid[t_i + 1, k] = (-2 * np.pi * (0 +1j ) * u_0 * k * del_t) * Cstar[t_i + 1, k] / L + Ck_grid[t_i, k]        
            
            change = (-2 * np.pi * 1j * u_0 * k * del_t) * Cstar[t_i + 1, k] / L        
            change_R = np.real(change * change.conjugate()) # obama
            change_grid[t_i, k] = change
            
### Runge Kutta
if scheme == 'SRK4':
    subtitle = 'spectral method, RK4'
    Z = 2 * np.pi * 1j* u_0 / L
    for t_i in np.arange(0, int(time/del_t) -1 , 1):
        a1 = Ck_grid[t_i,:] + Z * k * (Ck_grid[t_i, :] * del_t/2)
        a2 = Ck_grid[t_i,:] + Z * k * (a1 * del_t/2)
        a3 = Ck_grid[t_i,:] + Z * k * (a2 * del_t)
        a4 = Ck_grid[t_i,:] - Z * k * (a3 * del_t/2)
        
        Ck_grid[t_i +1, :] = (a1 + 2*a2 + a3 - a4)/3        
        

## plotje van het spectrum 
amps_final  = np.real(ma.masked_invalid(Ck_grid[-1, :]) * ma.masked_invalid(Ck_grid[-1, :].conjugate()))
 
plt.figure()
plt.title('Spectrum after int. ' + subtitle)
plt.plot(np.arange(0, int(n_x/2), 1), amps_final, linewidth=1.0, color='green')
plt.xlabel('k')
plt.ylabel('amplitude')
plt.savefig('figs/Spectrum after integr ' + subtitle)
plt.show()

############################ FT to real space ###############################

Ck_grid = ma.masked_invalid(Ck_grid)
C_end = np.zeros(n_x)

C_snapshots = np.zeros((len(range(0, 5000, 500)), J))

for x in range(n_x):
    C_end[x] = 0.0
    for k in np.arange(1, int(J/2), 1):
        C_end[x] =  C_end[x] + np.real( Ck_grid[-1,k] * (np.cos(2*np.pi*k*Lx[x]/L)) + 1j * np.sin(2*np.pi*k*Lx[x]/L))      
        
        
#%%
plt.figure()
plt.title("FT to realspace " + subtitle)

plt.plot(Lx/1000,C_end,linewidth=1.0, color = 'blue', label = 'final state')    
plt.plot(Lx/1000, C_init, linewidth = 1.0, color = 'red', label = 'initial state')
# normal settings   
# plt.text(100, 0.8, f'CFL condition: {CFL}')
# plt.text(100, 0.85, f'dx [km]: {del_x/1000}')
# plt.text(100, 0.9, f'dt [s]: {del_t}')

# special settings for SFE
plt.text(100, 140, f'CFL condition: {CFL}')
plt.text(100, 120, f'dx [km]: {del_x/1000}')
plt.text(100, 100, f'dt [s]: {del_t}')


plt.ylabel('amplitude')
plt.xlabel('Length domain [km]')

plt.legend()
plt.savefig('figs/FT to realspace'+subtitle)
plt.show()

##################### animation #############################################

        # fig = plt.figure()
        # ax = plt.axes(xlim=(L), ylim=(-1,1.5))
        # k = np.arange(0, 50)
            
        # #initial line, which are the initial values of eta
        # line, = ax.plot(k, Ck_grid[0,:])
        
        # def init():  # only required for bl  itting to give a clean slate.
        #     line.set_ydata([] * len(x)) #length of the line we want to plot, [] *len(x) will make an empty array[], of lenght x
        #     return line,
        
        # def animate(n):
        #     line.set_ydata(Ck_grid[n,:])  # update the data.
        #     ax.set_title('%03d'%(n))
        #     ax.plot(x, Ck_grid[0,:], color = 'red', ls = ':')
        #     if n == 101 or n == 201 or n == 301:
        #         tim.sleep(1)
        #     return line,
        
        # anim = FuncAnimation(fig, animate, init_func=init,interval=20, blit = False)
        
        # # Set up formatting for the movie files
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        
        # anim.save(f'Spectral_dt{del_t}_dx{del_x}.mp4', writer=writer)




