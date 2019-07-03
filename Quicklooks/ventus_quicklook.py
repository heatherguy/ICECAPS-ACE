#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:26:49 2019

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

# Ventus parsing

def ventus_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df[5] = df[5].astype(int)
        df['Date'] = pd.to_datetime(df[0]*10000000000+df[1]*100000000+df[2]*1000000+df[3]*10000+df[4]*100+df[5],format='%Y%m%d%H%M%S')
        df = df.set_index('Date')
        del df[0],df[1],df[2],df[3],df[4],df[5]
        df.columns = ['wsd', 'wdir', 'T', 'Checksum']
        df = df.sort_values('Date')
        new_idx = pd.date_range(pd.to_datetime(str(start),format='%y%m%d'),pd.to_datetime(str(stop),format='%y%m%d')+dt.timedelta(days=1),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        df= df.reindex(new_idx, fill_value=np.NaN)
    return df

def extract_ventus_data(start,stop,dpath,log_ventus):
    # Extract Ventus data into a pandas array
    # Data format:
    #<STX>SS.S DDD +TT.T xx*XX<CR><ETX>
    #        SS.S = wind speed (m/s)
    #        DDD = wind direction
    #        +TT.T = signed virtual temperature
    #        xx = status
    #        XX = checksum

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_ventus,'w')             # Open the log file for writing
    all_files = glob.glob('*.ventus*')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    v1 = pd.DataFrame()
    v2 = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty or contains non-ascii characters
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        # Filter and report files with non-ascii characters
        try:
            content = open(f).read()
        except:
            print('Data error with %s'%f)
            continue
        try:
            content.encode('ascii')
        except UnicodeDecodeError:
            log.write("Error with: %s contains non-ascii characters.\n"%f)  
            continue
            
        if f[-1]=='1':
            try:
                v1 = v1.append(pd.read_csv(f, header=None, delim_whitespace=True, error_bad_lines=False))
            except:
                print('Data error with %s'%f)
                continue
        if f[-1]=='2':
            try:
                v2 = v2.append(pd.read_csv(f, header=None, delim_whitespace=True, error_bad_lines=False))
            except:
                print('Data error with %s'%f)
                continue
        
    # Sort out the date referencing and columns
    v1 = ventus_pdf_sort(v1,start_f,stop_f)
    v2 = ventus_pdf_sort(v2,start_f,stop_f)
    log.write('Data parse finished\n')
    log.close()
    return v1,v2




# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_ventus =  '/home/fluxtower/Quicklooks/ventus_parse_log'

# For Licor and winds, plot one day previous
day_stop = dt.datetime.today()
day_start = day_stop - dt.timedelta(days=1)

# Plot set up for daily plots
myFmt = md.DateFormatter('%d-%H')
rule = md.HourLocator(interval=3)
minor = md.HourLocator(interval=1)
fig_size = (6,4)


# ventus data
print('Extracting Ventus data..')
v1,v2 = extract_ventus_data(day_start,day_stop,dpath,log_ventus)

# filter nans
v1 = v1.dropna()
v2 = v2.dropna()

if len(v1)!=0:
    print('Got V1 data')
else:
    print('!! No V1 data !!')
if len(v2)!=0:
    print('Got V2 data')
else:
    print('!! No V2 data !!')

v1.wdir = v1.wdir.astype(str)
v1 = v1[~v1.wdir.str.contains("F")]
v1.wdir = v1.wdir.astype(int)
v2.wdir = v2.wdir.astype(str)
v2 = v2[~v2.wdir.str.contains("F")]
v2.wdir = v2.wdir.astype(int)

v1.wsd = v1.wsd.astype(str)
v1.wsd = v1.wsd.str.lstrip('\x03\x02')
v1.wsd = v1.wsd.astype(float)
v2.wsd = v2.wsd.astype(str)
v2.wsd = v2.wsd.str.lstrip('\x03\x02')
v2.wsd = v2.wsd.astype(float)

 # ventus plot

fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(v1.index,v1.wsd,c='b',label='v1',alpha=0.5)
ax1.plot(v2.index,v2.wsd,c='r',label='v2',alpha=0.5)
ax1.grid('on')
ax1.set_ylabel('Wind Speed (m/s)')
ax1.legend(fontsize='x-small')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(v1.index,v1.wdir,c='b',alpha=0.5)
ax2.plot(v2.index,v2.wdir,c='r',alpha=0.5)
ax2.grid('on')
ax2.set_ylim(0,360)
ax2.set_yticks([0,90,180,270,360])
ax2.set_ylabel('Wind Direction')

# Format ticks and labels and layout
ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Ventus plot..')
fig.savefig('/home/fluxtower/Quicklooks/ventus_current.png')
fig.clf()






