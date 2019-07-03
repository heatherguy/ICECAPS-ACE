#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:48:45 2019

@author: heather
"""

# This script imports the raw CLASP data, processes the status parameter flags 
# and applies a flow correction to convert raw counts to concentrations in 
# particles per ml. 
# This script is based on the original CLASP processing MatLab scripts, by S. Norris.
# re-written in python for use at Summit Station Greenland. 
# Author: Heather Guy, 2019-04-30

# Import required packages
import numpy as np
import datetime as dt
import pandas as pd
from scipy import io

# Inputs
filename = '/Users/heather/Desktop/Summit_May_2019/Instruments/CLASP/Intercomparison_20190301/190301_160000.00.claspf'
channels = 16 # Number of aerosol concentration channels (usually 16)
calfile = '/Users/heather/Desktop/Summit_May_2019/Instruments/CLASP/CLASP-cal-Feb2019/calibration-unit-F-Feb2019.mat' # Calibration .mat file
sf = 1 # Sample frequency (hz)

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

# Function to convert interger to binary.
get_bin = lambda x, n: format(x, 'b').zfill(n)

# Loop through, extract and sort data into the dataframes initialised above
for i in range(0,len(data_block)):
    split = data_block[i].split()
    
    # Extract and store dates
    date = dt.datetime(int(split[0]),int(split[1]),int(split[2]),
                       int(split[3]),int(split[4]),
                       int(np.floor(float(split[5])))) 
    dates.append(date)
    
    # Extract and store counts
    counts = data_block[i][25:]
    for x in range(0,len(counts[14:].split())):
        CLASP[i,x] = float(counts[14:].split()[x])    
    
    # Extract, and convert staus addresses, store flags
    statusbyte=float(counts[0:3])
    binary=get_bin(int(statusbyte),8)
    statusaddr[i] = int(binary[4:8],2)
    heater[i] = int(binary[2])       # check you have these the right way around
         
    # Extract and store status parameters and overflow flag
    parameter[i]=float(counts[3:10])
    overflow[i]=float(counts[10:14])  
      
    # Check overflow flags and correct histogram. 
    # Overflow is for channels 1 to 8 only.
    for n in range(0,8):
        obin=get_bin(int(overflow[i]),8)
        if int(obin[::-1][0]):
            print('overfow recorded')
            CLASP[i,n] = CLASP[i,n] + 256

            
# Arrange parameters into a neat dataframe
param_len = len(statusaddr)/10
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

# Arrange Counts into a neat dataframe

CLASP_df = pd.DataFrame({'Date':dates, 'Heater flag':heater,
                        'c1 (n/s)':CLASP[:,0],'c2 (n/s)':CLASP[:,1],
                        'c3 (n/s)':CLASP[:,2],'c4 (n/s)':CLASP[:,3], 'c5 (n/s)':CLASP[:,4],
                        'c6 (n/s)':CLASP[:,5],'c7 (n/s)':CLASP[:,6],'c8 (n/s)':CLASP[:,7],'c9 (n/s)':CLASP[:,8],
                        'c10 (n/s)':CLASP[:,9],'c11 (n/s)':CLASP[:,10],'c12 (n/s)':CLASP[:,11], 
                        'c13 (n/s)':CLASP[:,12],'c14 (n/s)':CLASP[:,13],'c15 (n/s)':CLASP[:,14],'c16 (n/s)':CLASP[:,15]})
CLASP_df = CLASP_df.set_index('Date')
CLASP_df.index = pd.DatetimeIndex(CLASP_df.index)
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
        print('laser ref increased out of bounds')
    elif CLASP_df['LaserR'][i]<laser_ref -300:
        laser_flag.append(0)
        print('laser ref decreased out of bounds')
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

# Outputs: 
# Pandas data frame, all counts in n/s, all flags, all status parameters and calculated 
# sample volume. Concentrations can be calculated by counts/sample volume. 