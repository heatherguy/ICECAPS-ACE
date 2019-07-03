#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:45:01 2019

@author: heather
"""

# Import functions
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline
import numpy as np       
import datetime as dt    
import pylab as plb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import rc
from matplotlib import rcParams
import glob
import os
import itertools
import datetime
import pandas as pd

# Licor parsing

def extract_licor_data(start,stop,dpath,log_licor):

#2019 04 03 11 11 56.453 89	189	0.16469	35.4518	0.04404	297.105	20.74	99.0	1.5224
# Date, Ndx, DiagVal, CO2R, CO2D, H2OR, H2OD, T, P, cooler

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*.licor')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    licor = pd.DataFrame()

    # Function to convert to float otherwise write nan
    def convert_float(x):
        try:
            return np.float(x)
        except:
            return np.nan

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue 
        try:
            licor = licor.append(pd.read_csv(f, header=None, sep='\t',error_bad_lines=False))
        except:
            print('Bad data, skipped file %s'%f)
            continue
    
    # Sort out the date referencing and columns
    if licor.empty==False:
        licor['Date']=licor[0].str[0:23]
        licor = licor[licor['Date'].map(len) ==23]
        licor['Date'] = pd.to_datetime(licor['Date'],format='%Y %m %d %H %M %S.%f')
        licor['Ndx']=licor[0].str[24:].astype('int')
        licor['DiagV']=licor[1].astype('int')
        licor['CO2R'] = licor[2]
        licor['CO2R']=licor['CO2R'].apply(convert_float)
        licor['CO2D'] = licor[3]
        licor['CO2D']=licor['CO2D'].apply(convert_float)
        licor['H2OR'] = licor[4]
        licor['H2OR']=licor['H2OR'].apply(convert_float)
        licor['H2OD'] = licor[5]
        licor['H2OD']=licor['H2OD'].apply(convert_float)
        licor['T'] = licor[6].astype('float')
        licor['P'] = licor[7].astype('float')
        licor['cooler'] = licor[8].astype('float')
        del licor[0],licor[1],licor[2],licor[3],licor[4],licor[5],licor[6],licor[7],licor[8],licor[9]                 
        licor = licor.sort_values('Date')
        licor = licor.set_index('Date')
    else:
        log.write('No data from licor\n')
     
    log.write('Data parse finished\n')
    log.close()

    return licor


# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_licor =  '/home/fluxtower/Quicklooks/licor_parse_log'


# For Licor and winds, plot one day previous
day_stop = dt.datetime.today()
day_start = day_stop - dt.timedelta(days=1)

# Plot set up for daily plots
myFmt = md.DateFormatter('%d-%H')
rule = md.HourLocator(interval=3)
minor = md.HourLocator(interval=1)
fig_size = (6,4)


# Licor plot

# Licor data
#start=datetime.datetime(2019, 5, 19, 16, 49, 34, 873137)
#stop = datetime.datetime(2019, 5, 20, 16, 49, 34, 873137)
print('Extracting Licor data...')
licor = extract_licor_data(day_start,day_stop,dpath,log_licor)
if len(licor)!=0:
    print('Got Licor data')
    licor = licor[licor.H2OD>0]
    licor = licor[licor.H2OD<1000]
    licor = licor[licor.CO2D>0]
    licor = licor[licor.CO2D<1000]
else:
    print('!! No Licor data !!')

fig = plt.figure(figsize=fig_size)

#H2O density (mmol m-3)
ax1 = fig.add_subplot(211)
ax1.plot(licor.index,licor.H2OD,c='b',alpha=0.5)
ax1.grid('on')
ax1.set_ylabel('H2O density (mmol m-3)')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(licor.index,licor.CO2D,c='r',alpha=0.5)
ax2.grid('on')
ax2.set_ylabel('CO2 density (mmol m-3)')

# Format ticks and labels and layout
ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Licor plot...')
fig.savefig('/home/fluxtower/Quicklooks/licor_current.png')
fig.clf()

