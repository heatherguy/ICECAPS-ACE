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

        
    