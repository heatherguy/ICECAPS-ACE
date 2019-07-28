#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:37:40 2019

@author: heather
"""

# Import required packages

import matplotlib
matplotlib.use('Agg')

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
from matplotlib import rcParams
import os
import glob
from scipy import io
# Supress warnings for sake of log file
import warnings
warnings.filterwarnings("ignore")


# Function to save processed data

def clasp_process(filename,calfile,sf):
    # Read in the data
    fid = open(filename)
    data_block = list(filter(('\n').__ne__, fid.readlines()))
    fid.close()

    # Initialise empty dataframes
    dates = []
    CLASP = np.ones([len(data_block),16])*-999  # Counts
    statusaddr = np.ones(len(data_block))*-999  # Status address
    parameter = np.ones(len(data_block))*-999   # Parameter value
    overflow = np.ones(len(data_block))*-999    # Overflow (channel 1-8 only)
    #flow_check = np.ones(len(data_block))*-999  # True if flow is in range - this is too stringent, can ignore
    heater = np.ones(len(data_block))*-999      # True if heater is on
    #sync = np.ones(len(data_block))*-999       # CAN IGNORE THIS - it's not connected

    # Function to convery interger to binary.
    get_bin = lambda x, n: format(x, 'b').zfill(n)

    # Loop through, extract and sort data into the dataframes initialised above
    for i in range(0,len(data_block)):
        split = data_block[i].split()
        # Extract and store dates
        date = dt.datetime(int(split[0]),int(split[1]),int(split[2]),
                           int(split[3]),int(split[4]),
                           int(np.floor(float(split[5])))) 
        dates.append(date)   
        if len(split)!=25:
            continue
    
        # Extract and store counts
        for x in range(0,16):
            CLASP[i,x] = float(split[-17:-1][x])    
    
        # Extract, and convert staus addresses, store flags
        statusbyte=float(split[6])
        binary=get_bin(int(statusbyte),8)
        statusaddr[i] = int(binary[4:8],2)
        heater[i] = int(binary[2])       # check you have these the right way around
         
        # Extract and store status parameters and overflow flag
        parameter[i]=float(split[7])
        overflow[i]=float(split[8])  
      
        # Check overflow flags and correct histogram. 
        # Overflow is for channels 1 to 8 only.
        for n in range(0,8):
            obin=get_bin(int(overflow[i]),8)
            if int(obin[::-1][0]):
                #print('overfow recorded')
                CLASP[i,n] = CLASP[i,n] + 256
                
                
    # Arrange parameters into a neat dataframe
    param_dates = np.asarray(dates)[np.where(statusaddr==0)[0]]
    rejects = parameter[np.where(statusaddr==0)[0]]
    threshold = parameter[np.where(statusaddr==1)[0]]
    ThisFlow = parameter[np.where(statusaddr==2)[0]]
    FlowPWM = parameter[np.where(statusaddr==3)[0]]
    PumpCurrent = parameter[np.where(statusaddr==4)[0]]
    SensorT = parameter[np.where(statusaddr==5)[0]]
    HousingT = parameter[np.where(statusaddr==6)[0]]
    PumpT = parameter[np.where(statusaddr==7)[0]]
    SupplyV = parameter[np.where(statusaddr==8)[0]]
    LaserR  = parameter[np.where(statusaddr==9)[0]]
    paths=[param_dates,rejects,threshold,ThisFlow,FlowPWM,PumpCurrent,SensorT,HousingT,PumpT,SupplyV,LaserR]
    param_len = len(min(paths, key=len))

    param_df=pd.DataFrame({'Date':param_dates[0:param_len],'Rejects (n)':rejects[0:param_len],'Threshold (mV)':threshold[0:param_len],
                      'ThisFlow':ThisFlow[0:param_len],'FlowPWM':FlowPWM[0:param_len],'PumpCurrent (mA)':PumpCurrent[0:param_len],
                      'SensorT (C)':SensorT[0:param_len],'HousingT (C)':HousingT[0:param_len],'PumpT (C)':PumpT[0:param_len],
                      'SupplyV':SupplyV[0:param_len],'LaserR':LaserR[0:param_len]})

    param_df = param_df.set_index('Date')
    param_df.index = pd.DatetimeIndex(param_df.index)
    param_df = param_df.loc[~param_df.index.duplicated(keep='first')]

    # Arrange Counts into a neat dataframe

    CLASP_df = pd.DataFrame({'Date':dates, 'Heater flag':heater,
                        1:CLASP[:,0],2:CLASP[:,1],
                        3:CLASP[:,2],4:CLASP[:,3], 5:CLASP[:,4],
                        6:CLASP[:,5],7:CLASP[:,6],8:CLASP[:,7],9:CLASP[:,8],
                        10:CLASP[:,9],11:CLASP[:,10],12:CLASP[:,11], 
                        13:CLASP[:,12],14:CLASP[:,13],15:CLASP[:,14],16:CLASP[:,15]})

    CLASP_df = CLASP_df.set_index('Date')
    CLASP_df.index = pd.DatetimeIndex(CLASP_df.index)
    CLASP_df = CLASP_df.loc[~CLASP_df.index.duplicated(keep='first')]
    CLASP_df = pd.concat([CLASP_df, param_df], axis=1)

    # Apply flow corrections and quality flags, 
    # convert raw counts to concentrations in particles per ml.

    # Get calibration data
    # The data below are from calibration-unti-F-Feb2019.mat
    cal_dict=io.loadmat(calfile,matlab_compatible=True)
    channels = cal_dict['calibr'][0][0][3][0]        # array of AD channel number of boundaries of size 
    lowerR = cal_dict['calibr'][0][0][4][0]          # array of true radius of channels (micrometers)
    meanR = cal_dict['calibr'][0][0][5][0]           # array of mean radius (micrometers) of each size bin
    dr = cal_dict['calibr'][0][0][6][0]              # array of widths of each size bin (micrometers)
    setflow = cal_dict['calibr'][0][0][7][0][0]      # A2D value of flow monitor set as target flow
    TSIflow = cal_dict['calibr'][0][0][8][0]         # array of calibration flow rates from TSI flow meter
    realflow = cal_dict['calibr'][0][0][9][0]        # array of measured A2D flow rates matching TSflow
    laser_ref = cal_dict['calibr'][0][0][10][0][0]   # the laser reference voltage at time of calibration

    # Check the laser reference hasn't dropped off - laser flag good=1
    laser_flag = []
    for i in range(0,len(CLASP_df)):
        if CLASP_df['LaserR'][i]>laser_ref + 500:
            laser_flag.append(0)
            #p#rint('laser ref increased out of bounds')
        elif CLASP_df['LaserR'][i]<laser_ref -300:
            laser_flag.append(0)
            #print('laser ref decreased out of bounds')
        else:
            laser_flag.append(1)
        
    CLASP_df['laser flag']=laser_flag

    # Check the pump is working ok, good=1
    pump_flag = []
    for i in range(0,len(CLASP_df)):
        if CLASP_df['ThisFlow'][i]>setflow + 200:
            pump_flag.append(0)
            print('Pump flow increased out of bounds')
        elif CLASP_df['ThisFlow'][i]<setflow - 200:
            pump_flag.append(0)
            print('Pump flow decreased out of bounds')
        else:
            pump_flag.append(1)
        
    CLASP_df['pump flag']=pump_flag


    # Do flow correction and convert to concentations
    # TSI flow is from the TSI flowmeter, realflow is the flow the CLASP records internally

    P = np.polyfit(realflow,TSIflow,2) # These are from the flow calibration - fit a polynomial
    flow = np.polyval(P,CLASP_df['ThisFlow']) # flow in L/min
    flow_correction = ((flow/60)*1000)/sf # Sample volume in ml/s

    # Interpolate flow correction onto full timeseries and add to array
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x= nan_helper(flow_correction)
    flow_correction[nans]= np.interp(x(nans), x(~nans), flow_correction[~nans])
    CLASP_df['Sample volume (ml/s)']=flow_correction

    # Now to plot concentrations in counts/ml, just need to divide the counts/s by the sample volume
    # CLASP-G flowrate = 3L/minute = 50 cm3/second
    # Units: particle counts/ sample interval
    # Sample interval: 1s
    # Calculate total counts

    # Calculate concentation
    # sample volume. Concentrations can be calculated by counts/sample volume. 
    CLASP_df[CLASP_df.columns[0:16]]=CLASP_df[CLASP_df.columns[0:16]].div(CLASP_df['Sample volume (ml/s)'],axis=0)

    # Calculate total counts
    CLASP_df['CLASP_conc']=CLASP_df[1].astype(float)+ CLASP_df[2].astype(float)+CLASP_df[3].astype(float)+CLASP_df[4].astype(float)+CLASP_df[5].astype(float)+CLASP_df[6].astype(float)+CLASP_df[7].astype(float)+CLASP_df[8].astype(float)+CLASP_df[9].astype(float)+CLASP_df[10].astype(float)+CLASP_df[11].astype(float)+CLASP_df[12].astype(float)+CLASP_df[13].astype(float)+CLASP_df[14].astype(float)+CLASP_df[15].astype(float)+CLASP_df[16].astype(float)
    #CLASP_df['CLASP_conc'] = CLASP_df['total_counts'] / flowrate #counts/cm3

    # Outputs: 
    # Pandas data frame, all counts in n/s, all flags, all status parameters and calculated 
    # sample volume. Concentrations can be calculated by counts/sample volume. 


    # Save dataframe as text file
    CLASP_df.to_csv(opath+'Processed_'+fname,sep=' ',na_rep='-999')
    return



# Function to get distribution
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

# Get Matlab processed clasp file
def get_matlab_clasp(d_loc,fname):
    df = pd.read_csv(d_loc+fname,sep=' ',header=None)
    return df
    
# Get python processed clasp file
def get_python_clasp(d_loc,fname):
    df = pd.read_csv(d_loc+fname,sep=' ',index_col=0,parse_dates=True,na_values=-999)
    counts = df[df.columns[0:16]]
    counts = counts.apply(pd.to_numeric, errors='coerce')
    df.sort_values(by=['Date'], inplace=True)
    counts.sort_values(by=['Date'], inplace=True)
    return df, counts

# Plot aerosol size spectra
def plot_spectra(in_dists,labels):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.grid(True)
    for i in range(0,len(in_dists)):
        ax.loglog(in_dists[i][0],in_dists[i][1],label=labels[i])

    ax.set_xlabel('Diameter (d) / $\mu$m')
    ax.set_ylabel('dN/dlogd (cc$^{-3}$)')
    #ax.set_ylim(100000,100000000)
    ax.legend(loc='best',fontsize=10)
    fig.tight_layout()
    #return fig

def plot_clasp_params(df):
    df.dropna(inplace=True)
    params = df.columns[17:-5].to_list()
    fig = plt.figure(figsize=[25,25])
    for n in range(0,len(params)):
        if n<5:
            ax = plt.subplot2grid((5,2),(n,0),fig=fig)
            ax.plot(df.index,df[params[n]],label=params[n])
            ax.legend(loc='best',fontsize=15)
            ax.grid('on')
        elif n<11:
            ax = plt.subplot2grid((5,2),(n-5,1),fig=fig)
            ax.plot(df.index,df[params[n]],label=params[n])
            ax.legend(loc='best',fontsize=15)
            ax.grid('on')
    fig.tight_layout()
    #return fig
    
# Get between dates.
def get_between_dates_clasp(d_loc,d1,d2,name):
    os.chdir(d_loc)                  # Change directory to where the data is
    all_clasp_files = glob.glob('*%s*'%name)  # List all data files
    # Extract daterange
    clasp_file_dates = np.asarray([dt.datetime.strptime(f[-14:-4],"%Y-%m-%d") for f in all_clasp_files])
    clasp_idxs = np.where(np.logical_and(clasp_file_dates>=d1, clasp_file_dates<=d2))[0]
    clasp_dfs = [all_clasp_files[i] for i in clasp_idxs]
    all_dfs=[]
    counts_dfs=[]
    for fname in clasp_dfs:
        all_dfs.append(get_python_clasp(d_loc,fname)[0])
        counts_dfs.append(get_python_clasp(d_loc,fname)[1])

    df = pd.concat(all_dfs)
    counts = pd.concat(counts_dfs)
    
    df.sort_values(by=['Date'], inplace=True)
    counts.sort_values(by=['Date'], inplace=True)
    
    return df, counts

def plot_clasp_ts(df,d1,d2,name):
    fig = plt.figure(figsize=[12,4])
    myFmt = md.DateFormatter('%d/%H')
    ax1 = fig.add_subplot(111)
    ax1.semilogy(df.index,df.CLASP_conc,label='%s'%name,alpha=0.6)
    ax1.grid('on')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0.01,100000)
    ax1.set_ylabel('Total Particle Counts / cm$^3$')
    ax1.set_title('Total Particle Counts: %s to %s'%(d1.date(),d2.date()))
    ax1.set_xlim(d1,d2)
    ax1.xaxis.set_major_formatter(myFmt)
    fig.tight_layout()
    #return fig