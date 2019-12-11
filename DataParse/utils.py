#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:39:27 2019

@author: heather
"""

# Import things
import matplotlib
#matplotlib.use('Agg')

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
from matplotlib import rcParams
import os
import glob
from scipy import io
import tarfile
import re

# Supress warnings for sake of log file
import warnings
warnings.filterwarnings("ignore")


# Extract files containting 'instr' from tar.gz files and save in outdir.
def extract_tar(dloc,outdir,instr):
    fnames = glob.glob(dloc + r'*.tar.gz')
    for f in fnames: 
        t = tarfile.open(f,'r')
        mems = t.getmembers()
        for m in mems:
            if instr in m.name:
                t.extractall(path=outdir, members=[m])
        t.close()
        
        
# Plot aerosol size distribution spectra
def get_dist(df,nbins,bounds):
    if len(bounds)!=nbins+1:
        print('Error bounds')
        return

    mid_points = [(bounds[i+1]+bounds[i])/2 for i in range(0,nbins)]
    logd = np.log(bounds)   
    dlogd = [logd[i+1]-logd[i] for i in range(0,len(mid_points))]
    # Sum columns
    hist = df.sum(axis=0)
    dNdlogd = hist/dlogd
    return mid_points,dNdlogd

def plot_dist(dists,labels,xlims):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.grid(True)
    for i in range(0,len(dists)):
        ax.loglog(dists[i][0],dists[i][1],label=labels[i])

    ax.set_xlim(xlims[0],xlims[1])
    ax.set_xlabel('Diameter (d) / $\mu$m')
    ax.set_ylabel('dN/dlogd (cc$^{-3}$)')
    ax.legend(loc='best',fontsize=10)
    fig.tight_layout()
    # return fig or save fig.


    
# Get NOAA weather data
#Fields: Wind D, Wind s (m/s), Wind steadiness, pressure (hPa), 2 m T, 10 m T, tower T, RH, P (mm/hr)
# Missing values: -999, -999.9, -9, -999.90, -999, -999, -999, -99, -99

def get_NOAA_met(w_dloc):
    all_dates = []
    all_data =  []
    f = open(w_dloc,mode='r')
    data = f.read().split('\n')
    for i in range(0,len(data)):
        if data[i][4:20]=='':
            continue       
        else:
            all_dates.append(pd.to_datetime(data[i][4:20],format='%Y %m %d %H %M'))
            all_data.append(list(map(float, data[i][20:].split())))

    wd = [int(x[0]) for x in all_data]
    ws = [x[1] for x in all_data]
    p = [x[3] for x in all_data]
    T = [x[4] for x in all_data]
    RH = [int(x[7]) for x in all_data]

    # Mask missing values

    wd = np.ma.masked_where(np.asarray(wd)==-999, np.asarray(wd))
    ws = np.ma.masked_where(np.asarray(ws)==-999.9, np.asarray(ws))
    p = np.ma.masked_where(np.asarray(p)==-999.9, np.asarray(p))
    T = np.ma.masked_where(np.asarray(T)==-999.9, np.asarray(T))
    RH = np.ma.masked_where(np.asarray(RH)==-99, np.asarray(RH))

    # Create pandas dataframe
    d = {'date' : all_dates,'wd' : wd, 'ws' : ws, 'pres' : p, 'T' : T, 'RH' : RH}
    pdf = pd.DataFrame(d)
    pdf['date'] = pd.to_datetime(pdf.date)
    pdf = pdf.sort_values(by='date')
    pdf = pdf.set_index('date')

    # Delete duplicates and fill any missing data with blank lines   
    sd = min(all_dates)
    ed = max(all_dates)
    d_list = pd.date_range(sd, ed,freq='Min')
    pdf = pdf[~pdf.index.duplicated(keep='first')]
    pdf = pdf.reindex(d_list)
    
    return pdf
        
    