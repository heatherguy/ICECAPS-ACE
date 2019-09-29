#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:37:57 2019

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
import io
import pandas as pd

# HMP155 parsing

def HMP_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df = df.dropna()
        df['Second'] = df['Second'].astype(float)
        df['Second'] = df['Second'].astype(int)
        df['Minute'] = df['Minute'].astype(int)
        df['Hour'] = df['Hour'].astype(int)
        df['Day'] = df['Day'].astype(int)
        df['Month'] = df['Month'].astype(int)
        df['Year'] = df['Year'].astype(int)
        df['Date'] = pd.to_datetime(df['Year']*10000000000+df['Month']*100000000+df['Day']*1000000+df['Hour']*10000+df['Minute']*100+df['Second'],format='%Y%m%d%H%M%S')
        df = df.set_index('Date')
        del df['Year'],df['Month'],df['Day'],df['Hour'],df['Minute'],df['Second'],df['junk']
        df.columns = ['RH', 'Ta', 'Tw', 'Err', 'h']
        df['RH']=df['RH'].astype(float)
        df['Tw']=df['Tw'].astype(float)
        df['Ta']=df['Ta'].astype(float)
        #df['h']=df['h'].astype(int)       
        df = df.sort_values('Date')
        new_idx = pd.date_range(pd.to_datetime(start).round('1s'),pd.to_datetime(stop).round('1s'),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        #df= df.reindex(new_idx, fill_value=np.NaN)
    else:
        df = pd.DataFrame(columns=['RH', 'Ta', 'Tw', 'Err', 'h'])
    return df

def extract_HMP_data(name, start,stop,dpath,logf):
    # Extract HMP155 data into a pandas array
    # Data format: YYYY MM DD HH MM.mmm TT:TT:TT RH Ta Tw Err hs
    # Ta = seperate probe T, Tw = wetbulb t, hs = heating status

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(logf,'w')             # Open the log file for writing
    all_files = glob.glob('*.%s'%name)  # List all data files
    
    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    HMP = pd.DataFrame()

    # Extract the data
    for f in dfs:
        # Ignore file if it's empty 
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        fd = io.open(f,"r",errors='replace')
        f_dat = fd.readlines()
        clean_dat = [i for i in f_dat if len(i)>=60 and len(i)<=63]
        pdf = pd.DataFrame(clean_dat)
        pdf[1] = pdf[0].str.split()
        final_df = pd.DataFrame(pdf[1].values.tolist(), columns=['Year','Month','Day','Hour','Minute','Second','junk','RH','Ta', 'Tw', 'Err', 'h'])
        # Store good data
        HMP = HMP.append(final_df)
        
 
    # Sort out the date referencing and columns
    HMP = HMP_pdf_sort(HMP,start,stop)

    log.write('Data parse finished\n')
    log.close()
    return HMP


# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_hmp = '/home/fluxtower/Quicklooks/HMP_parse_log'
#dpath = '/Users/heather/ICECAPS-ACE/Data'
#log_hmp = '/Users/heather/ICECAPS-ACE/Quicklooks/HMP_parse_log'

# For KT, SND: Plot one week previous from now
week_stop = dt.datetime.today()
week_start = week_stop - dt.timedelta(days=7)

# Plot set up for weekly plots
myFmt = md.DateFormatter('%b %d')
rule = md.DayLocator(interval=1)
minor = md.HourLocator(interval=6)
fig_size = (6,4)

# HMP data
print('Extracting HMP data')
HMP1 = extract_HMP_data('HMP1',week_start,week_stop,dpath,log_hmp)
print('Got HMP1')
HMP2 = extract_HMP_data('HMP2',week_start,week_stop,dpath,log_hmp)
print('Got HMP2')
HMP3 = extract_HMP_data('HMP3',week_start,week_stop,dpath,log_hmp)
print('Got HMP3')
HMP4 = extract_HMP_data('HMP4',week_start,week_stop,dpath,log_hmp)
print('Got HMP4')
print('Building plot')

# Temperature/ humidity plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(HMP1.index,HMP1.Ta,c='b',label='HMP1',alpha=0.8)
ax1.plot(HMP2.index,HMP2.Ta,c='r',label='HMP2',alpha=0.8)
ax1.plot(HMP3.index,HMP3.Ta,c='g',label='HMP3',alpha=0.8)
ax1.plot(HMP4.index,HMP4.Ta,c='m',label='HMP4',alpha=0.8)
ax1.grid('on')
ax1.set_ylabel(u'T \N{DEGREE SIGN}C')
ax1.legend(fontsize='xx-small')
ax1.set_xlim(week_start,week_stop)


ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(HMP1.index,HMP1.RH,c='b',label='HMP1',alpha=0.8)
ax2.plot(HMP2.index,HMP2.RH,c='r',label='HMP2',alpha=0.8)
ax2.plot(HMP3.index,HMP3.RH,c='g',label='HMP3',alpha=0.8)
ax2.plot(HMP4.index,HMP4.RH,c='m',label='HMP4',alpha=0.8)
ax2.set_ylabel(u'RH %')
ax2.grid('on')
ax2.legend(fontsize='xx-small')

# Format ticks and labels and layout
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)

fig.tight_layout()
print('Saving HMP plot')
fig.savefig('/home/fluxtower/Quicklooks/T_RH_current.png')
#fig.savefig('/Users/heather/ICECAPS-ACE/Quicklooks/T_RH_current.png')

fig.clf()
