#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:32:23 2019

@author: heather
"""

import matplotlib
#matplotlib.use('Agg')
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
from matplotlib import rcParams
import matplotlib.colors as colors
import os
import glob
from scipy import io

# Get CPC data
def get_cpc(d_loc,d1,d2):
    os.chdir(d_loc)                  # Change directory to where the data is
    #log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*CPC*')
    file_dates = np.asarray([(dt.datetime.strptime(f[-14:-4], '%Y-%m-%d')).date() for f in all_files]) 
    idxs = np.where(np.logical_and(file_dates>=d1.date(), file_dates<=d2.date()))[0]
    dfs = [all_files[i] for i in idxs]
    cpc = pd.DataFrame()
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            #log.write('Error with: '+f+' this file is empty.\n')
            continue 
        cpc = cpc.append(pd.read_csv(f,sep=',',error_bad_lines=False,header=None,parse_dates={'Dates' : [0,1,2,3,4,5]}))  

    cpc.Dates = pd.to_datetime(cpc.Dates,format='%Y %m %d %H %M %S')
    cpc = cpc.sort_values('Dates')
    cpc = cpc.set_index(cpc['Dates'])
    cpc.index = pd.DatetimeIndex(cpc.index)
    del cpc['Dates']
    cpc_counts =cpc.rename(columns={6:'Concentration (/cm3)'})
    return cpc_counts

## Get OPC data
    
def get_opc(opc_n,d_loc,d1,d2):
    os.chdir(d_loc)                  # Change directory to where the data is
    #log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*%s*OPC*'%opc_n)
    if opc_n=='TAWO':
        file_dates = np.asarray([(dt.datetime.strptime(f[-12:-4], '%Y%m%d')).date() for f in all_files])
    else:
        file_dates = np.asarray([(dt.datetime.strptime(f[-14:-4], '%Y-%m-%d')).date() for f in all_files])
           
    idxs = np.where(np.logical_and(file_dates>=d1.date(), file_dates<=d2.date()))[0]
    dfs = [all_files[i] for i in idxs]
    opc = pd.DataFrame()
    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            #log.write('Error with: '+f+' this file is empty.\n')
            continue 
        opc = opc.append(pd.read_csv(f, skiprows=4,sep=',',error_bad_lines=False))  
    opc['Dates'] = pd.to_datetime(opc['time'],format='%Y-%m-%d %H:%M:%S')
    opc = opc.sort_values('Dates')
    opc = opc.set_index(opc['Dates'])
    opc.index = pd.DatetimeIndex(opc.index)
    #opc = opc[~opc.index.duplicated()]
    del opc['time'], opc['Dates']

    # Convert flow rate from L/min to cm3/s
    # 1 L/min = 16.66667 cm3/s
    opc.FlowRate = opc.FlowRate/100 * 16.66667

    opc_counts = opc[opc.columns[0:24]]
    opc_counts = opc_counts.apply(pd.to_numeric, errors='coerce')
    opc_params = opc[opc.columns[24:]]
    
    # Convert counts/interval to total counts/s
    opc.period = opc.period/100.0 # period in s
    opc_counts = opc_counts.divide(opc.period, axis=0)
    # Convert total counts/second to counts/cm3
    opc_counts = opc_counts.divide(opc.FlowRate, axis=0)

    return opc_counts, opc_params

# Function to read SKYOPC data
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

# Function to read and import GRIMM OPC data
def get_skyopc(d_loc,d1,d2,save=False):
    os.chdir(d_loc)                  # Change directory to where the data is
    #log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*SKYOPC*')
    file_dates = np.asarray([(dt.datetime.strptime(f[-14:-4], '%Y-%m-%d')).date() for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=d1.date(), file_dates<=d2.date()))[0]
    dfs = [all_files[i] for i in idxs]
    skyopc = pd.DataFrame()
    c=np.nan
    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            #log.write('Error with: '+f+' this file is empty.\n')
            continue 
        f_data = open(f)
        d = f_data.readlines()
        f_data.close()
        for i in range(0,len(d)):
            line=d[i].split()
            if line[0] =='P':
                if len(line)!=17:
                    c=0
                    datetime=np.nan
                    continue
                #Year Mon Day Hr Min Loc 4Tmp Err pA/p pR/p UeL Ue4 Ue3 Ue2 Ue1 Iv 
                datetime = dt.datetime(int(line[1])+2000,int(line[2]),int(line[3]),int(line[4]),int(line[5]))
                #datetime = dt.datetime.strptime('20'+line[1]+line[2]+line[3]+line[4]+line[5],'%Y%m%d%H%M')
                quad_Tmp = int(line[7])
                Err = int(line[8])
                pAp = int(line[9])
                pRp = int(line[10])
                Int = int(line[16])
                c=0
            
            elif len(line)!=9:
                continue

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
                if isinstance(datetime,dt.datetime):
                    skyopc = skyopc.append(pd.Series([datetime+dt.timedelta(seconds=n*6), ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19, ch20, ch21, ch22, ch23, ch24, ch25, ch26, ch27, ch28, ch29, ch30, ch31, ch32, quad_Tmp,Err,pAp,pRp,Int]),ignore_index=True)
            
    # remove repeated channel 16
    del skyopc[16]
    # Correct counts for size bins 'all counts above lower threshold.'        
    for i in range(2,16):
        skyopc[i-1]=skyopc[i-1]-skyopc[i]
    for i in range(18,33):
        skyopc[i-1]=skyopc[i-1]-skyopc[i]
    
    skyopc=skyopc.rename(columns={0: 'Date',1:'ch1' ,2: 'ch2', 3: 'ch3',4: 'ch4',5: 'ch5',6: 'ch6',7: 'ch7',8: 'ch8',9: 'ch9',10: 'ch10',11: 'ch11',12: 'ch12',13: 'ch13',14: 'ch14',15: 'ch15',16: 'ch16',17: 'ch17',18: 'ch18',19: 'ch19', 20:'ch20',21: 'ch21',22: 'ch22',23: 'ch23',24: 'ch24',25: 'ch25',26: 'ch26',27: 'ch27',28: 'ch28',29: 'ch29',30: 'ch30',31: 'ch31',32: 'ch32',33: 'quad_Tmp',34:'Err',35:'pAp',36:'pRp',37:'Int'})
    skyopc.dropna(inplace=True)
    skyopc = skyopc.set_index('Date')
    skyopc = skyopc.sort_values('Date')
    skyopc.index = pd.DatetimeIndex(skyopc.index)
    skyopc = skyopc[~skyopc.index.duplicated()]
    
    skyopc_counts = skyopc[skyopc.columns[0:31]]
    skyopc_counts = skyopc_counts.apply(pd.to_numeric, errors='coerce') # Counts in counts/ 6 seconds
    skyopc_counts =skyopc_counts / 100.0 # convert from counts/100ml to counts/cm3    
    skyopc_params = skyopc[skyopc.columns[31:]]
    
    if save: 
        skyopc_counts.to_csv(save+'SKYOPC_counts_%s'%(str(d1.date())))
        skyopc_params.to_csv(save+'SKYOPC_params_%s'%(str(d1.date())))
    
    return skyopc_counts, skyopc_params

# Function to read and process CLASP data
# Inputs

def get_clasp(d_loc,d1,d2,claspn,channels,calfile,sf):
    # Function to convery interger to binary.
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    os.chdir(d_loc)                  # Change directory to where the data is
    #log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*%s*'%claspn)
    file_dates = np.asarray([(dt.datetime.strptime(f[-14:-4], '%Y-%m-%d')).date() for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=d1.date(), file_dates<=d2.date()))[0]
    dfs = [all_files[i] for i in idxs]
    data_block=[]
    for f in dfs: 
        # Read in the data
        fid = open(f)
        data_block.append(list(filter(('\n').__ne__, fid.readlines())))
        fid.close()
        
    data_block=list(np.concatenate(data_block))
     
    # Initialise empty dataframes
    dates = []
    CLASP = np.ones([np.shape(data_block)[0],16])*-999  # Counts
    statusaddr = np.ones(np.shape(data_block)[0])*-999  # Status address
    parameter = np.ones(np.shape(data_block)[0])*-999   # Parameter value
    overflow = np.ones(np.shape(data_block)[0])*-999    # Overflow (channel 1-8 only)
    #flow_check = np.ones(len(data_block))*-999  # True if flow is in range - this is too stringent, can ignore
    heater = np.ones(np.shape(data_block)[0])*-999      # True if heater is on
    #sync = np.ones(len(data_block))*-999       # CAN IGNORE THIS - it's not connected

    # Loop through, extract and sort data into the dataframes initialised above
    for i in range(0,np.shape(data_block)[0]):
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
            CLASP[i,x] = float(split[-16:][x])    
    
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
    param_len = int(len(statusaddr)/10)
    
    rejects = parameter[np.where(statusaddr==0)[0][0:param_len]]
    threshold = parameter[np.where(statusaddr==1)[0][0:param_len]]
    ThisFlow = parameter[np.where(statusaddr==2)[0][0:param_len]]
    FlowPWM = parameter[np.where(statusaddr==3)[0][0:param_len]]
    PumpCurrent = parameter[np.where(statusaddr==4)[0][0:param_len]]
    SensorT = parameter[np.where(statusaddr==5)[0][0:param_len]]
    HousingT = parameter[np.where(statusaddr==6)[0][0:param_len]]
    PumpT = parameter[np.where(statusaddr==7)[0][0:param_len]]
    SupplyV = parameter[np.where(statusaddr==8)[0][0:param_len]]
    LaserR  = parameter[np.where(statusaddr==9)[0][0:param_len]]
    param_dates = np.asarray(dates)[np.where(statusaddr==0)[0][0:param_len]]
   
    if len(param_dates)<param_len:
        param_len = len(param_dates)
    if len(rejects)<param_len:
        param_len = len(rejects)
    if len(threshold)<param_len:
        param_len = len(threshold)
    if len(ThisFlow)<param_len:
        param_len = len(ThisFlow)
    if len(FlowPWM)<param_len:
        param_len = len(FlowPWM)
    if len(PumpCurrent)<param_len:
        param_len = len(PumpCurrent)
    if len(SensorT)<param_len:
        param_len = len(SensorT)
    if len(HousingT)<param_len:
        param_len = len(HousingT)
    if len(PumpT)<param_len:
        param_len = len(PumpT)
    if len(SupplyV)<param_len:
        param_len = len(SupplyV)
    if len(LaserR)<param_len:
        param_len = len(LaserR)

    param_dates = np.asarray(dates)[np.where(statusaddr==0)[0][0:param_len]] 
    rejects = parameter[np.where(statusaddr==0)[0][0:param_len]]
    threshold = parameter[np.where(statusaddr==1)[0][0:param_len]]
    ThisFlow = parameter[np.where(statusaddr==2)[0][0:param_len]]
    FlowPWM = parameter[np.where(statusaddr==3)[0][0:param_len]]
    PumpCurrent = parameter[np.where(statusaddr==4)[0][0:param_len]]
    SensorT = parameter[np.where(statusaddr==5)[0][0:param_len]]
    HousingT = parameter[np.where(statusaddr==6)[0][0:param_len]]
    PumpT = parameter[np.where(statusaddr==7)[0][0:param_len]]
    SupplyV = parameter[np.where(statusaddr==8)[0][0:param_len]]
    LaserR  = parameter[np.where(statusaddr==9)[0][0:param_len]]

    param_df=pd.DataFrame({'Date':param_dates,'Rejects (n)':rejects,'Threshold (mV)':threshold,
                        'ThisFlow':ThisFlow,'FlowPWM':FlowPWM,'PumpCurrent (mA)':PumpCurrent,
                        'SensorT (C)':SensorT,'HousingT (C)':HousingT,'PumpT (C)':PumpT,
                        'SupplyV':SupplyV,'LaserR':LaserR})

    param_df = param_df.set_index('Date')
    param_df.index = pd.DatetimeIndex(param_df.index)
    param_df = param_df[~param_df.index.duplicated()]

    # Arrange Counts into a neat dataframe
    CLASP_df = pd.DataFrame({'Date':dates,
                        1:CLASP[:,0],2:CLASP[:,1],
                        3:CLASP[:,2],4:CLASP[:,3], 5:CLASP[:,4],
                        6:CLASP[:,5],7:CLASP[:,6],8:CLASP[:,7],9:CLASP[:,8],
                        10:CLASP[:,9],11:CLASP[:,10],12:CLASP[:,11], 
                        13:CLASP[:,12],14:CLASP[:,13],15:CLASP[:,14],16:CLASP[:,15]})
    CLASP_df = CLASP_df.set_index('Date')
    CLASP_df.index = pd.DatetimeIndex(CLASP_df.index)
    CLASP_df = CLASP_df[~CLASP_df.index.duplicated()]
    CLASP_df = pd.concat([CLASP_df, param_df], axis=1)

    # Apply flow corrections and quality flags, 
    # convert raw counts to concentrations in particles per ml.
    # Get calibration data
    cal_dict=io.loadmat(calfile,matlab_compatible=True)
    TSIflow = cal_dict['calibr'][0][0][8][0]         # array of calibration flow rates from TSI flow meter
    realflow = cal_dict['calibr'][0][0][9][0]        # array of measured A2D flow rates matching TSflow

    # Do flow correction and convert to concentations
    # TSI flow is from the TSI flowmeter, realflow is the flow the CLASP records internally
    P = np.polyfit(realflow,TSIflow,2) # These are from the flow calibration - fit a polynomial
    flow = np.polyval(P,CLASP_df['ThisFlow']) # flow in L/min
    flow_correction = ((flow/60)*1000)/sf # Sample volume in cm3/s

    # Interpolate flow correction onto full timeseries and add to array
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x= nan_helper(flow_correction)
    flow_correction[nans]= np.interp(x(nans), x(~nans), flow_correction[~nans])
    CLASP_df['Sample volume (ml/s)']=flow_correction
    

    # Now to plot concentrations in counts/cm3, just need to divide the counts/s by the sample volume
    clasp_counts = CLASP_df[CLASP_df.columns[0:16]]
    clasp_params = CLASP_df[CLASP_df.columns[16:]]
    clasp_counts = clasp_counts.apply(pd.to_numeric, errors='coerce')
    clasp_counts = clasp_counts.divide(flow_correction, axis=0)
    
    # CLASP-G flowrate = 3L/minute = 50 cm3/second
    # Units: particle counts/ sample interval
    # Sample interval: 1s
    # Calculate total counts
    # Calculate concentation
    
    #CLASP_df['Concentration (/cm3)'] = CLASP_df['total_counts'] / 50 #counts/cm3
    
    return clasp_counts,clasp_params

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

