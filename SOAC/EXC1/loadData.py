#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:24:25 2020

@author: ivo
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as P

path = r'/home/ivo/SOAC/EXC1/IjmuidenData.csv' 
data = pd.read_csv(path)

dpdx = data['dpdx[Pa/km]'] /1000

time = np.arange(0,48,1) #hrs

def pres_grad_func(t, A, B, phi):
    omega = 0.000072792
    return A * np.cos(omega * t*3600 + phi) + B

params, covs = curve_fit(pres_grad_func, time, dpdx, p0 =[0.001, 0.001 ,5], maxfev = 10000)

print('Found fit has parameters A = {}, B = {}, phi = {}'.format(params[0], params[1], params[2]))

P.plot(time, dpdx)
P.plot(time, pres_grad_func(time, *params))































# def load_data(path):
#     df = pd.read_csv(path, skiprows=31, low_memory=0)
#     time = pd.to_datetime(df['YYYYMMDD'])
#     df['YYYYMMDD'] = time
#     return df

# df_ijmuiden = load_data(r'/home/ivo/SOAC/EXC1/ijmuiden.txt')
# df_valkenburg = load_data(r'/home/ivo/SOAC/EXC1/valkenburg.txt')
# df_schiphol = load_data(r'/home/ivo/SOAC/EXC1/schiphol.txt')