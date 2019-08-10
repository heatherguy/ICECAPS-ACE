#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:23:17 2019

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


# Metek parsing

def metek_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df['ms']= df[5]*1000000
        df['ms']= df['ms'].astype(int)
        df[0] = df[0].astype(str)
        df[1] = df[1].astype(str)
        df[2] = df[2].astype(str)
        df[3] = df[3].astype(str)
        df[4] = df[4].astype(str)
        df['ms'] = df['ms'].astype(str)        
        df['Date'] = pd.to_datetime(df[0]+df[1]+df[2]+df[3]+df[4]+df['ms'],format='%Y%m%d%H%M%f')
        df = df.set_index('Date')
        del df[0],df[1],df[2],df[3],df[4],df[5],df['ms'],df[7],df[9],df[10],df[12],df[13],df[15],df[16]
        df.columns = ['Status','x','y','z','T']
        df = df[pd.to_numeric(df['T'], errors='coerce').notnull()]        
        df['T']=df['T'].astype(float)
        df['T']=df['T']/100
        df = df.sort_values('Date')
        #new_idx = pd.date_range(pd.to_datetime(str(start),format='%y%m%d'),pd.to_datetime(str(stop),format='%y%m%d'),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        #df= df.reindex(new_idx, fill_value=np.NaN)
    return df

def extract_metek_data(start,stop,dpath,log_metek):
    # Extract metek data into a pandas array
    # Data format:
    #2019 04 02 16 23 41.734 M:x =    14 y =    -1 z =    12 t =  2357
    # M = measured data heater off
    # H = measured data heater on
    # D = measured data heater defect
    # x,y,z componenets of wind in cm/s
    # t = acoustic temperature 2357 = 23.57C
    
    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_metek,'w')             # Open the log file for writing
    all_files = glob.glob('*.metek*')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    m1 = pd.DataFrame()
    m2 = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        if f[-1]=='1':
            try:
                m1 = m1.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False))
            except:
                skiprows=1
                datapass=0
                while datapass==0:
                    try:
                        m1 = m1.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False,skiprows=skiprows))
                        datapass=1
                    except:
                        skiprows=skiprows+1

        if f[-1]=='2':
            try:
                m2 = m2.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False))
            except:
                skiprows=1
                datapass=0
                while datapass==0:
                    try:
                        m2 = m2.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False,skiprows=skiprows))
                        datapass=1
                    except:
                        skiprows=skiprows+1
               
        
    # Sort out the date referencing and columns
    m1 = metek_pdf_sort(m1,start_f,stop_f)
    m2 = metek_pdf_sort(m2,start_f,stop_f)
    log.write('Data parse finished\n')
    log.close()
    return m1,m2    

# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_metek =  '/home/fluxtower/Quicklooks/metek_parse_log'
#dpath = '/Users/heather/ICECAPS-ACE/Data/'
#log_metek =  '/Users/heather/ICECAPS-ACE/Quicklooks/metek_parse_log'

# For Licor and winds, plot one day previous
day_stop = dt.datetime.today()
day_start = day_stop - dt.timedelta(days=1)
fig_size = (6,4)

# Plot set up for daily plots
myFmt = md.DateFormatter('%d-%H')
rule = md.HourLocator(interval=3)
minor = md.HourLocator(interval=1)

# metek data
print('Extracting Metek Data...')
m1,m2 = extract_metek_data(day_start,day_stop,dpath,log_metek)
# remove poor quality data.
m1.x = pd.to_numeric(m1.x,errors='coerce')
m1.y = pd.to_numeric(m1.y,errors='coerce')
m1.z = pd.to_numeric(m1.z,errors='coerce')
if len(m1)!=0:
    print('Got m1 data')
else:
    print('!! No m1 data !!')

m2.x = pd.to_numeric(m2.x,errors='coerce')
m2.y = pd.to_numeric(m2.y,errors='coerce')
m2.z = pd.to_numeric(m2.z,errors='coerce')
if len(m2)!=0:
    print('Got m2 data')
else:
    print('!! No m2 data !!')

# Metek plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(m1.index,m1.x/100,c='r',label='x',alpha=0.5)
ax1.plot(m1.index,m1.y/100,c='b',label='y',alpha=0.5)
ax1.plot(m1.index,m1.z/100,c='g',label='z',alpha=0.5)
ax1.axhline(0,c='k',lw=1)
ax1.grid('on')
ax1.set_ylabel('M1 wind speed, m/s')
ax1.legend(fontsize='xx-small')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(m2.index,m2.x/100,c='r',label='x',alpha=0.5)
ax2.plot(m2.index,m2.y/100,c='b',label='y',alpha=0.5)
ax2.plot(m2.index,m2.z/100,c='g',label='z',alpha=0.5)
ax2.axhline(0,c='k',lw=1)
ax2.grid('on')
ax2.set_ylabel('M2 wind speed, m/s')
ax2.legend(fontsize='x-small')

# Format ticks and labels and layout
ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Metek plot...')
fig.savefig('/home/fluxtower/Quicklooks/metek_current.png')
fig.clf()










