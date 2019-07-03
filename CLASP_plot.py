# -*- coding: utf-8 -*-
"""
@author: guyh
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np       
import datetime as dt    
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import rcParams
import pandas as pd

# Supress warnings for sake of log file
import warnings
warnings.filterwarnings("ignore")

# Plotting preferences

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 14}) 
rcParams['axes.titlepad'] = 14 
rcParams['xtick.major.pad']='10'
rcParams['ytick.major.pad']='10'
myFmt = md.DateFormatter('%H')
rule = md.HourLocator(interval=1)
 

# Get plot times
today = dt.datetime.utcnow()
today = today.replace(minute=0, second=0, microsecond=0)
yesterday = today - dt.timedelta(hours=24)


# Get data

yfname = '/home/fluxtower/Data/CLASP_F_Summit_%s.csv'%dt.datetime.strftime(yesterday,'%Y-%m-%d')
tfname = '/home/fluxtower/Data/CLASP_F_Summit_%s.csv'%dt.datetime.strftime(today,'%Y-%m-%d')

f1 = open(yfname)
d = f1.readlines()
data_block1 = list(filter(('\n').__ne__, d))
f1.close()
f2 = open(tfname)
d = f2.readlines()
data_block2 = list(filter(('\n').__ne__, d))
f2.close()
data_block = np.concatenate([data_block1,data_block2])
#ata_block = data_block2

dates = []
data = np.ones([len(data_block),16])*-999
status_parameter = np.ones(len(data_block))*-999
parameter_val = np.ones(len(data_block))*-999
ascii_val = np.ones(len(data_block))*-999

for i in range(0,len(data_block)):
    split = data_block[i].split('\t')
    date = dt.datetime(int(split[0]),int(split[1]),int(split[2]),int(split[3]),int(split[4]),int(split[5])) 
    dates.append(date)
    counts = split[6][:-1]
    status_parameter[i]=float(counts[0:4])
    parameter_val[i]=float(counts[4:10])
    ascii_val[i]=float(counts[10:14])  
    for x in range(0,len(counts[14:].split())):
        data[i,x] = float(counts[14:].split()[x])


# Plot & save

bins = [0.18,	0.200,	0.250,	0.300,	0.350,	0.400,	0.450,	0.500,	0.750,	1.000,	1.500,	2.000,	2.500,	3.000,	3.500,	4.500,	5.500]
# mask any -999s
data = np.ma.masked_where(data ==-999, data)

y_lims = [1,16]
if today.hour!=0:     
    fig = plt.figure(figsize=(17,5))
    ax = fig.add_subplot(111)
    x_lims = [md.date2num(today - dt.timedelta(hours=24)),md.date2num(today)]
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.)
    cs = plt.pcolormesh(dates,np.arange(0,16,1), np.transpose(data),vmin=0,vmax=200)
    cb = plt.colorbar(cs,extend='max',label='Counts/second',orientation='horizontal',pad=0.18,aspect=50,shrink=0.7)
    ax.xaxis_date()
    ax.set_xlim(x_lims)
    ax.set_title('CLASP: %s'%((dt.datetime.strftime(yesterday,'%Y-%m-%d')+' to '+dt.datetime.strftime(today,'%Y-%m-%d'))))
    ax.set_ylabel('Particle Size ($\mu$m)')
    ax.set_yticks(np.arange(0,16,1))
    ax.set_yticklabels(bins)
    ax.set_xlabel('Hours (UTC)')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(rule)   
    fig.tight_layout()
    fig.savefig('/home/fluxtower/Data/CLASP_current.png')
else:
    fig = plt.figure(figsize=(17,5))
    ax = fig.add_subplot(111)
    cmap = matplotlib.cm.viridis
    cmap.set_bad('white',1.)
    cs = plt.pcolormesh(dates,np.arange(0,16,1), np.transpose(data),vmin=0,vmax=200)
    cb = plt.colorbar(cs,extend='max',label='Counts/second',orientation='horizontal',pad=0.18,aspect=50,shrink=0.7)
    ax.xaxis_date()
    ax.set_xlim(yesterday, today)
    ax.set_title('CLASP: %s'%dt.datetime.strftime(yesterday,'%Y-%m-%d'))
    ax.set_ylabel('Particle Size ($\mu$m)')
    ax.set_yticks(np.arange(0,16,1))
    ax.set_yticklabels(bins)
    ax.set_xlabel('Hours (UTC)')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(rule)   
    fig.tight_layout()
    fig.savefig('/home/fluxtower/Data/CLASP_plot_archive/CLASP_%s.png'%dt.datetime.strftime(yesterday,'%Y-%m-%d'))
    
fig.clf()