#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:56:19 2019

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

# KT15 parsing function
def extract_KT_data(start,stop,dpath,logf):
    # Extract KT15 data into a pandas array
    # Data format: YYYY MM DD HH MM.mmm TT.tt C
    # TT.tt = temperature, C = celcius

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(logf,'w')             # Open the log file for writing
    all_files = glob.glob('*.KT15')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    KT = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty or contains non-ascii characters
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue

        # Filter and report files with non-ascii characters
        content = open(f).read()
        try:
            content.encode('ascii')
        except UnicodeDecodeError:
            log.write("Error with: %s contains non-ascii characters.\n"%f)  
            continue
        
        # Store good data      
        KT = KT.append(pd.read_csv(f, header=None, delim_whitespace=True))
       
    # Sort out the date referencing and columns
    if KT.empty==False:
        KT[5] = KT[5].astype(int)
        KT['Date'] = pd.to_datetime(KT[0]*10000000000+KT[1]*100000000+KT[2]*1000000+KT[3]*10000+KT[4]*100+KT[5],format='%Y%m%d%H%M%S')
        KT = KT.set_index('Date')
        del KT[0],KT[1],KT[2],KT[3],KT[4],KT[5]
        KT.columns = ['T', 'Units']
        KT = KT.sort_values('Date')
        new_idx = pd.date_range(pd.to_datetime(str(start_f),format='%y%m%d'),pd.to_datetime(str(stop_f),format='%y%m%d')+dt.timedelta(days=1),freq='1s' )
        KT.index = pd.DatetimeIndex(KT.index)
        KT = KT[~KT.index.duplicated()]
        KT= KT.reindex(new_idx, fill_value=np.NaN)
        log.write('Data parse finished\n')
    else:
        log.write('No KT data found for this period')
    
    log.close()
    return KT

# SnD parsing

def extract_snd_data(start,stop,dpath,log_snd):
#aa;D.DDD;QQQ; VVVVV;CC
#aa = 33 (serial address of sensor)
#D.DDD = Distance to target in m (will need temperature adjustment
#QQQ = Data quality, varies beteen 152-600, 600 is the poorest quality
#VVVVV = diagnostic tests (only first two are actually something), 1 = pass. 
#CC = two-character checksum of data packet. (indication of data errors? Not sure how to read this. )
    
    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_snd,'w')             # Open the log file for writing
    all_files = glob.glob('*.SnD')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    snd = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        snd = snd.append(pd.read_csv(f, header=None, delim_whitespace=True, error_bad_lines=False))
        
    if snd.empty==False:
        snd.dropna(inplace=True)
        snd[5] = snd[5].astype(int)
        snd['Date'] = pd.to_datetime(snd[0]*10000000000+snd[1]*100000000+snd[2]*1000000+snd[3]*10000+snd[4]*100+snd[5],format='%Y%m%d%H%M%S')
        snd = snd.set_index('Date')
        new = snd[6].str.split(';',expand=True)
        snd['depth'] = new[1]
        snd['Q']=new[2]
        snd['V']=new[3]
        snd['C']=new[4]
        del snd[0],snd[1],snd[2],snd[3],snd[4],snd[5],snd[6]
        snd = snd.sort_values('Date')
        snd.depth = snd.depth.astype(float)
        new_idx = pd.date_range(pd.to_datetime(str(start_f),format='%y%m%d'),pd.to_datetime(str(stop_f),format='%y%m%d')+dt.timedelta(days=1),freq='1s' )
        snd.index = pd.DatetimeIndex(snd.index)
        snd = snd[~snd.index.duplicated()]
        snd= snd.reindex(new_idx, fill_value=np.NaN)   
    else: 
        log.write('No data from snd\n')
       
    log.write('Data parse finished\n')
    log.close()

    return snd


# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data'
log_KT = '/home/fluxtower/Quicklooks/KT_parse_log'
log_snd =  '/home/fluxtower/Quicklooks/snd_parse_log'

# For KT, SND: Plot one week previous from now
week_stop = dt.datetime.today()
week_start = week_stop - dt.timedelta(days=7)

# Plot set up for weekly plots
myFmt = md.DateFormatter('%b %d')
rule = md.DayLocator(interval=1)
minor = md.HourLocator(interval=6)
fig_size = (6,4)

# KT data and plot
print('Extracting Data from KT15...')
KT = extract_KT_data(week_start,week_stop,dpath,log_KT)
dropna_KT = KT.dropna(subset=['T'])
if len(dropna_KT)!=0:
    print('Got KT15 data')
else: 
    print('!! No KT15 data !!')
    
# SnD data and plot
print('Extracting Data from SnD...')
snd = extract_snd_data(week_start,week_stop,dpath,log_snd)
dropna_snd = snd.dropna(subset=['depth'])
if len(dropna_snd)!=0:
    print('Got SnD data')
else:
    print('!! No SnD data !!')


# Snow surface (T and depth) plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(dropna_KT.index,dropna_KT['T'])
ax1.grid('on')
ax1.set_ylabel(u'Snow Temperature \N{DEGREE SIGN}C')
ax1.set_xlim(week_start,week_stop)
ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(dropna_snd.index,dropna_snd['depth'])
ax2.grid('on')
ax2.set_ylim(0,2)
ax2.set_ylabel(u'Distance from sensor (m)')
# Snd status check.
#ax1.text('SnD Status')

ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)

fig.tight_layout()
print('Saving snow surface plot.')
fig.savefig('/home/fluxtower/Quicklooks/snow_current.png')
fig.clf()
