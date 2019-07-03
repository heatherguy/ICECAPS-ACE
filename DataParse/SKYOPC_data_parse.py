#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:00:08 2019

@author: heather
"""

# Import functions and set up plots

import numpy as np       
import datetime as dt    
import pylab as plb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib import rc
from matplotlib import rcParams
import glob
import itertools
import pandas as pd

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 12}) 
myFmt = md.DateFormatter('%H:%M')
rule = md.MinuteLocator(interval=15)

SKYOPC_dfile = '/Users/heather/Desktop/MSF_aerosol_intercomparison/SKYOPC_Summit_2019-05-31.csv'

# Get SKYOPC Data
# Measurement interval 6 seconds
# I think C0 = time
# C1 = time + 6 s
# C2 = time + 12 s
# ect.

# 32 channels 
# data output in the unit particle/100ml

# SKYOPC chaneel boundaries:
#0.25,0.28,0.3,0.35,0.4,0.45,0.5,0.58,0.65,0.7,0.8,1.0,1.3,1.6,2,2.5,3,3.5,4,5,6.5,7.5,8.5,10,12.5,15,17.5,20,25,30,32 
#channels 16 and 17 are identical (overlapping 
#channel for different physical measurement ranges)...so one should be 
#discarded before analysis.

#Note also that I think these are diameters not radii - the manual 
#doesn't say!...but in discussion of the calibration it refers only to 
#particle diameter.



# Function to read and import GRIMM OPC data
def read_skyopc(fname):
    f = open(fname)
    d = f.readlines()
    f.close()
    for i in range(0,len(d)):
        line=d[i].split()
        if len(line)<8:
            continue
        if line[0] =='P':
            #Year Mon Day Hr Min Loc 4Tmp Err pA/p pR/p UeL Ue4 Ue3 Ue2 Ue1 Iv 
            datetime = dt.datetime.strptime('20'+line[1]+line[2]+line[3]+line[4]+line[5],'%Y%m%d%H%M')
            quad_Tmp = int(line[7])
            Err = int(line[8])
            pAp = int(line[9])
            pRp = int(line[10])
            Int = int(line[16])
            c=0
        
        elif c==0: 
            ch1=int(line[1])
            ch2=int(line[2])
            ch3=int(line[3])
            ch4=int(line[4])
            ch5=int(line[5])
            ch6=int(line[6])
            ch7=int(line[7])
            ch8=int(line[8])
            c = c+1    
        elif c ==1:
            ch9=int(line[1])
            ch10=int(line[2])
            ch11=int(line[3])
            ch12=int(line[4])
            ch13=int(line[5])
            ch14=int(line[6])
            ch15=int(line[7])
            ch16=int(line[8])
            c = c+1
        elif c == 2:
            ch17=int(line[1])
            ch18=int(line[2])
            ch19=int(line[3])
            ch20=int(line[4])
            ch21=int(line[5])
            ch22=int(line[6])
            ch23=int(line[7])
            ch24 =int(line[8])
            c= c+1
        elif c==3:
            ch25=int(line[1])
            ch26=int(line[2])
            ch27=int(line[3])
            ch28=int(line[4])
            ch29=int(line[5])
            ch30=int(line[6])
            ch31=int(line[7])
            ch32=int(line[8])
            c = 0
            n = int(line[0][-2])
            if i<5:
                GRIMM_data = np.array([datetime+dt.timedelta(seconds=n*6), ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21, ch22, ch23, ch24, ch25, ch26, ch27, ch28, ch29, ch30, ch31, ch32, quad_Tmp,Err,pAp,pRp,Int])
            else:
                GRIMM_data = np.vstack((GRIMM_data,np.array([datetime+dt.timedelta(seconds=n*6), ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21, ch22, ch23, ch24, ch25, ch26, ch27, ch28, ch29, ch30, ch31, ch32, quad_Tmp,Err,pAp,pRp,Int])))

    return GRIMM_data

skyopc_data = read_skyopc(SKYOPC_dfile)
skyopc = pd.DataFrame(data=skyopc_data,columns=['Date','ch1' ,'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16', 'ch17', 'ch18', 'ch19', 'ch20', 'ch21', 'ch22', 'ch23', 'ch24', 'ch25', 'ch26', 'ch27', 'ch28', 'ch29', 'ch30', 'ch31', 'ch32', 'quad_Tmp','Err','pAp','pRp','Int'])
skyopc.dropna(inplace=True)
skyopc = skyopc.set_index('Date')
skyopc = skyopc.sort_values('Date')
skyopc.index = pd.DatetimeIndex(skyopc.index)
skyopc = skyopc[~skyopc.index.duplicated()]
# Change hour to an hour behind
skyopc.index = skyopc.index - pd.Timedelta(hours=1)
# remove repeated channel 16
del skyopc['ch16']

# SKYOPC
# Units: counts/100ml == 100 counts/cm3
# Calculate total counts/cm3 by adding bins
skyopc['total_counts']=skyopc['ch1']+skyopc['ch2']+skyopc['ch3']+skyopc['ch4']+skyopc['ch5']+skyopc['ch6']+skyopc['ch7']+skyopc['ch8']+skyopc['ch9']+skyopc['ch10']+skyopc['ch11']+skyopc['ch12']+skyopc['ch13']+skyopc['ch14']+skyopc['ch15']+skyopc['ch17']+skyopc['ch18']+skyopc['ch19']+skyopc['ch20']+skyopc['ch21']+skyopc['ch22']+skyopc['ch23']+skyopc['ch24']+skyopc['ch25']+skyopc['ch26']+skyopc['ch27']+skyopc['ch28']+skyopc['ch29']+skyopc['ch30']+skyopc['ch31']+skyopc['ch32']
# Assuming 6 second integration
skyopc['total_counts']=skyopc['total_counts']/100 #counts/cm3
skyopc['total_counts']=skyopc['total_counts'].astype(float)
