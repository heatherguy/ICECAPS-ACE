#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:25:53 2019

@author: heather
"""
import numpy as np
import datetime as dt
import pandas as pd
from scipy import io

from CLASP_functions import *


# Process CLASP data
# Inputs
dpath = '/Users/heather/ICECAPS-ACE/temp_data/home/fluxtower/Data/'
opath = '/Users/heather/Desktop/CLASP_troubleshoot_201900727/processed/'
#flowrate = 50 #3L/minute = 50 cm3/second
channels = 16 # Number of aerosol concentration channels (usually 16)
calfile = '/Users/heather/Desktop/Summit_May_2019/Instruments/CLASP/CLASP-cal-Feb2019/calibration-unit-G-Feb2019.mat' # Calibration .mat file
sf = 1 # Sample frequency (hz)

d1 = dt.datetime(2019,7,26)
d2 = dt.datetime(2019,7,27)
date_list = pd.date_range(d1,d2).to_list()

#for i in range(0,len(date_list)):
#    fname = 'CLASP_G_Summit_%s.csv'%str(date_list[i].date())
#    filename = dpath+fname
#    try: 
#        clasp_process(filename,calfile,sf)
#    except:
#        continue
    
d_loc='/Users/heather/Desktop/CLASP_troubleshoot_201900727/processed/'    

# Plot spectra
nbins=16
bounds=[0.3,0.4,0.5,0.6,0.7,1,2,3,4,5,6,7,8,9,10,12,14]
name='CLASP_F'
df = get_between_dates_clasp(d_loc,d1,d2,name)[1]
dist = get_dist(df,nbins,bounds)
plot_spectra([dist],[name])


# Plot python processed time series (between d1 and d2)
df = get_between_dates_clasp(d_loc,d1,d2,name)[0]
plot_clasp_ts(df,d1,d2,name)

# Plot Clasp params.
plot_clasp_params(df)

