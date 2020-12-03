
"""
Created on Wed Nov 25 10:25:58 2020

@author: ivo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time as tim

time = 4* 250. * 1e3 # s 
del_t = 2500. # s


L= 2500. * 1e3 # m
del_x = 25. * 1e3 # m

u_0 = 10. # m/s
J = 100 

C_0 = 1. # arbitrary

C_grid = np.zeros((int(time/del_t)+1, J))
C_grid[0,45:56] = C_0

for t_index, t in enumerate(np.arange(0, time, del_t)):
   
    for x_index, x in enumerate(np.arange(0, L, del_x)):
       C_grid[t_index+1, x_index] = -u_0 * (del_t/del_x) * (C_grid[t_index, x_index] - C_grid[t_index, x_index-1]) + C_grid[t_index, x_index]

#%%
fig = plt.figure()
ax = plt.axes(xlim=(L), ylim=(-1,1.5))
x = np.arange(0,L,del_x)
    
#initial line, which are the initial values of eta
line, = ax.plot(x, C_grid[0,:])

def init():  # only required for bl  itting to give a clean slate.
    line.set_ydata([] * len(x)) #length of the line we want to plot, [] *len(x) will make an empty array[], of lenght x
    return line,

def animate(n):
    line.set_ydata(C_grid[n,:])  # update the data.
    ax.set_title('%03d'%(n))
    ax.plot(x, C_grid[0,:], color = 'red', ls = ':')
    if n == 101 or n == 201 or n == 301:
        tim.sleep(1)
    return line,

anim = FuncAnimation(fig, animate, init_func=init,interval=20, blit = False)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim.save(f'Euler_dt{del_t}_dx{del_x}.mp4', writer=writer)




