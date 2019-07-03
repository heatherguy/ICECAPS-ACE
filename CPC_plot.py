
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


# Function to get just the last lines of a file

def tail( f, lines=20 ):
    total_lines_wanted = lines
    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = [] # blocks of size BLOCK_SIZE, in reverse order starting
                # from the end of the file
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            # read the last block we haven't yet read
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            # file too small, start from begining
            f.seek(0,0)
            # only read what was not read
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    all_read_text = all_read_text.decode("utf-8")
    return '\n'.join(all_read_text.splitlines()[-total_lines_wanted:])

# Get plot times
today = dt.datetime.utcnow()
today = today.replace(minute=0, second=0, microsecond=0)
yesterday = today - dt.timedelta(hours=24)

# Get CPC data

cpc_yfname = '/home/fluxtower/Data/CPC_Summit_%s.csv'%dt.datetime.strftime(yesterday,'%Y-%m-%d')
cpc_tfname = '/home/fluxtower/Data/CPC_Summit_%s.csv'%dt.datetime.strftime(today,'%Y-%m-%d')

cpc_data1 = np.genfromtxt(cpc_yfname,delimiter=',', missing_values='', usemask=True)
cpc_data2 = np.genfromtxt(cpc_tfname,delimiter=',', missing_values='', usemask=True)


# Check incase CPC is not running
if len(cpc_data1) == 0:
    if len(cpc_data2)==0:
        cpc_data = np.zeros([0,7])
    else:
        cpc_data = cpc_data2
elif len(cpc_data2)==0:
    cpc_data = cpc_data1
else:
    cpc_data = np.concatenate([cpc_data1,cpc_data2])
    
cpc_count = cpc_data[:,6]
cpc_dates = []
for i in range(0,len(cpc_data)):
    cpc_dates.append(dt.datetime(int(cpc_data[i,0]),int(cpc_data[i,1]),int(cpc_data[i,2]),int(cpc_data[i,3]),int(cpc_data[i,4]),int(cpc_data[i,5])))


# Get CLASP data
clasp_yfname = '/home/fluxtower/Data/CLASP_F_Summit_%s.csv'%dt.datetime.strftime(yesterday,'%Y-%m-%d')
clasp_tfname = '/home/fluxtower/Data/CLASP_F_Summit_%s.csv'%dt.datetime.strftime(today,'%Y-%m-%d')
clasp_f1 = open(clasp_yfname)
clasp_d = clasp_f1.readlines()
clasp_data_block1 = list(filter(('\n').__ne__, clasp_d))
clasp_f1.close()
clasp_f2 = open(clasp_tfname)
clasp_d = clasp_f2.readlines()
clasp_data_block2 = list(filter(('\n').__ne__, clasp_d))
clasp_f2.close()
clasp_data_block = np.concatenate([clasp_data_block1,clasp_data_block2])
clasp_dates = []
clasp_data = np.ones([len(clasp_data_block),16])*-999

for i in range(0,len(clasp_data_block)):
    split = clasp_data_block[i].split('\t')
    date = dt.datetime(int(split[0]),int(split[1]),int(split[2]),int(split[3]),int(split[4]),int(split[5])) 
    clasp_dates.append(date)
    clasp_counts = split[6][:-1]
    for x in range(0,len(clasp_counts[14:].split())):
        clasp_data[i,x] = float(clasp_counts[14:].split()[x])

all_clasp_counts = np.sum(clasp_data,axis=1)
clasp_ts = pd.Series(all_clasp_counts, index=clasp_dates)


# Get OPC data

OPCf =  '/home/fluxtower/Data/OPC/opc_current.dat'
opc_f = open(OPCf,mode='r+b')
opc_data = tail(opc_f,lines=1440).split('\n')
# If the data file has been restarted. 
if len(opc_data)<1440:
    missing = 1440 - len(opc_data)
    i=0
    while i<missing:
        opc_data.insert(1, [])
        i = i+1   
opc_dates = []
opc_counts = np.ones([len(opc_data),16])*-999# 16 bins
for i in range(0,len(opc_data)):
    add_min=0
    if opc_data[i]==[]:
        opc_dates.append(yesterday+dt.timedelta(minutes=add_min))
        add_min=add_min+1       
        continue
    elif opc_data[i].split(',')[0][:-3]=='t':
        opc_dates.append(yesterday)
        continue
    else:
        opc_dates.append(dt.datetime.strptime(opc_data[i].split(',')[0][:-3],'%Y-%m-%d %H:%M'))
        opc_counts[i,:] = opc_data[i].split(',')[1:17] 
# mask any -999s
opc_counts = np.ma.masked_where(opc_counts<0, opc_counts)
all_opc_counts = np.sum(opc_counts,axis=1)

# Set up y lim
max_counts = max([max(cpc_count),max(all_opc_counts),max(all_clasp_counts)])
if max_counts<100:
    yulim = 100
else:
    yulim = max_counts+10


# Plot & save

if today.hour != 0: 
    fig = plt.figure(figsize=(17,4))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.semilogy(cpc_dates,cpc_count, label='CPC (>5nm)',zorder=3,alpha=0.8)
    ax.semilogy(clasp_ts,label='CLASP (0.18-5.5$\mu$m)',zorder=2,alpha=0.8)
    ax.semilogy(opc_dates,all_opc_counts,label = 'OPC (0.38-17$\mu$m)',zorder=1)  
    ax.set_ylim(0,yulim)
    ax.set_ylabel('Total Particle Counts / cm$^3$')
    ax.set_title('Total Particle Counts: %s'%((dt.datetime.strftime(yesterday,'%Y-%m-%d')+' to '+dt.datetime.strftime(today,'%Y-%m-%d'))))
    ax.set_xlabel('Hours (UTC)')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(rule)
    ax.set_xlim(yesterday,today)
    ax.legend(loc='best',fontsize=10)
    fig.tight_layout()
    fig.savefig('/home/fluxtower/Data/CPC_current.png')
else:            
    fig = plt.figure(figsize=(17,4))
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.semilogy(cpc_dates,cpc_count, label='CPC (>5nm)',zorder=3,alpha=0.8)
    ax.semilogy(clasp_ts,label='CLASP (0.18-5.5$\mu$m)',zorder=2,alpha=0.8)
    ax.semilogy(opc_dates,all_opc_counts,label = 'OPC (0.38-17$\mu$m)',zorder=1)
    ax.set_ylim(0,yulim)
    ax.set_ylabel('Total Particle Counts / cm$^3$')
    ax.set_title('Total Particle Counts: %s'%dt.datetime.strftime(yesterday,'%Y-%m-%d'))
    ax.set_xlabel('Hours (UTC)')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(rule)
    ax.set_xlim(yesterday,today)
    ax.legend(loc='best',fontsize=10)
    fig.tight_layout()
    fig.savefig('/home/fluxtower/Data/CPC_plot_archive/CPC_%s.png'%dt.datetime.strftime(yesterday,'%Y-%m-%d'))

fig.clf()