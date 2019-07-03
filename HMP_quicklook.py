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
import pandas as pd

# HMP155 parsing

def HMP_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
	if df.empty==False:
		df.dropna(inplace=True)
		df[5] = df[5].astype(int)
		df['Date'] = pd.to_datetime(df[0]*10000000000+df[1]*100000000+df[2]*1000000+df[3]*10000+df[4]*100+df[5],format='%Y%m%d%H%M%S')
		df = df.set_index('Date')
		del df[0],df[1],df[2],df[3],df[4],df[5],df[6]
		df.columns = ['RH', 'Ta', 'Tw', 'Err', 'h']
		df = df.sort_values('Date')
		new_idx = pd.date_range(pd.to_datetime(str(start),format='%y%m%d'),pd.to_datetime(str(stop),format='%y%m%d')+dt.timedelta(days=1),freq='1s' )
		df.index = pd.DatetimeIndex(df.index)
		df = df[~df.index.duplicated()]
		df= df.reindex(new_idx, fill_value=np.NaN)
	else:
        	df = pd.DataFrame(columns=['RH', 'Ta', 'Tw', 'Err', 'h'])
	return df

def extract_HMP_data(start,stop,dpath,logf):
    # Extract HMP155 data into a pandas array
    # Data format: YYYY MM DD HH MM.mmm TT:TT:TT RH Ta Tw Err hs
    # Ta = seperate probe T, Tw = wetbulb t, hs = heating status

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(logf,'w')             # Open the log file for writing
    all_files = glob.glob('*.HMP*')  # List all data files
    
    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    HMP1 = pd.DataFrame()
    HMP2 = pd.DataFrame()
    HMP3 = pd.DataFrame()
    HMP4 = pd.DataFrame()
    HMP5 = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty or contains non-ascii characters
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue

        # Store good data for HMP1-5      
        if f[-1]=='1':
            try:
                HMP1 = HMP1.append(pd.read_csv(f, header=None, encoding='utf-8', delim_whitespace=True, error_bad_lines=False))
            except:
                continue
        elif f[-1]=='2':
            try: 
                HMP2 = HMP2.append(pd.read_csv(f, header=None, encoding='utf-8', delim_whitespace=True, error_bad_lines=False))
            except:
                continue
        elif f[-1]=='3':
            try:
                HMP3 = HMP3.append(pd.read_csv(f, header=None, encoding='utf-8', delim_whitespace=True, error_bad_lines=False))
            except:
                continue
        elif f[-1]=='4':
            try:
                HMP4 = HMP4.append(pd.read_csv(f, header=None, encoding='utf-8', delim_whitespace=True, error_bad_lines=False))
            except:
                continue
        elif f[-1]=='5':
            try:
                HMP4 = HMP5.append(pd.read_csv(f, header=None, encoding='utf-8', delim_whitespace=True, error_bad_lines=False))
            except:
                continue
        else:
            log.write('Error with %s, file name error.\n'%f)

        
    # Sort out the date referencing and columns
    HMP1 = HMP_pdf_sort(HMP1,start_f,stop_f)
    HMP2 = HMP_pdf_sort(HMP2,start_f,stop_f)
    HMP3 = HMP_pdf_sort(HMP3,start_f,stop_f)
    HMP4 = HMP_pdf_sort(HMP4,start_f,stop_f)
    HMP5 = HMP_pdf_sort(HMP5,start_f,stop_f)
    log.write('Data parse finished\n')
    log.close()
    return HMP1,HMP2,HMP3,HMP4,HMP5




# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 9}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_hmp = '/home/fluxtower/Quicklooks/HMP_parse_log'

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
HMP1,HMP2,HMP3,HMP4,HMP5 = extract_HMP_data(week_start,week_stop,dpath,log_hmp)

# Temperature/ humidity plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(HMP1.index,HMP1.Ta,c='b',label='HMP1',alpha=0.8)
ax1.plot(HMP2.index,HMP2.Ta,c='r',label='HMP2',alpha=0.8)
ax1.plot(HMP3.index,HMP3.Ta,c='g',label='HMP3',alpha=0.8)
ax1.plot(HMP4.index,HMP4.Ta,c='m',label='HMP4',alpha=0.8)
#ax1.plot(m1.index,m1['T'],c='c',label='m1')
#ax1.plot(m2.index,m2['T'],c='k',label='m2')
#ax1.plot(v1.index,v1['T'],c='orange',label='v1')
#ax1.plot(v2.index,v2['T'],c='grey',label='v2')
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
fig.clf()
