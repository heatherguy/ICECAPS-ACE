
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
import pandas as pd
from matplotlib import rcParams
import os
import glob
from scipy import io

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

#d_loc='/Users/heather/Desktop/aerosol_quicklooks/'
d_loc = '/home/fluxtower/'
calfile = '/home/fluxtower/CLASP-ICECAP-ACE/CLASP-cal-Feb2019/calibration-unit-F-Feb2019.mat' 
#calfile = '/Users/heather/Desktop/Summit_May_2019/Instruments/CLASP/CLASP-cal-Feb2019/calibration-unit-F-Feb2019.mat' # Calibration .mat file


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
yesterday = today - dt.timedelta(hours=24)

#today = dt.datetime(2019,7,17,14,0)
#yesterday = dt.datetime(2019,7,16,14,0)

# Get CPC data

cpc_yfname = d_loc + 'Data/CPC_Summit_%s.csv'%dt.datetime.strftime(yesterday,'%Y-%m-%d')
cpc_tfname = d_loc + 'Data/CPC_Summit_%s.csv'%dt.datetime.strftime(today,'%Y-%m-%d')

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




## Get OPC data
    
def get_opc(opc_n,d_loc,d1,d2):
    os.chdir(d_loc+'Data/')                  # Change directory to where the data is
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

    # Get total counts
    opc['total_counts']=opc['b0'].astype(float)+ opc['b1'].astype(float)+ opc['b2'].astype(float)+ opc['b3'].astype(float)+ opc['b4'].astype(float)+ opc['b5'].astype(float)+ opc['b6'].astype(float)+ opc['b7'].astype(float)+ opc['b8'].astype(float)+ opc['b9'].astype(float)+ opc['b10'].astype(float)+ opc['b11'].astype(float)+ opc['b12'].astype(float)+ opc['b13'].astype(float)+ opc['b14'].astype(float)+ opc['b15'].astype(float)+ opc['b16'].astype(float)+ opc['b17'].astype(float)+ opc['b18'].astype(float)+ opc['b19'].astype(float)+ opc['b20'].astype(float)+ opc['b21'].astype(float)+ opc['b22'].astype(float)+ opc['b23'].astype(float)
    opc['total_counts']=opc['total_counts'].replace({0: np.nan})
    # Convert total counts/interval to total counts/s
    opc.period = opc.period/100 # period in s
    opc.total_counts = opc.total_counts /opc.period
    # Convert total counts/second to counts/cm3
    opc['OPC_conc'] = opc.total_counts / opc.FlowRate

    return opc

try:
    TAWO_OPC = get_opc('TAWO',d_loc,yesterday,today)
except:
    print('No TAWO OPC data')

try:
    MSF_OPC = get_opc('MSF',d_loc,yesterday,today)
except:
    print('No MSF OPC data')

######################################################################################

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
def get_skyopc(d_loc,d1,d2):
    os.chdir(d_loc+'Data/')                  # Change directory to where the data is
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
            

    skyopc=skyopc.rename(columns={0: 'Date',1:'ch1' ,2: 'ch2', 3: 'ch3',4: 'ch4',5: 'ch5',6: 'ch6',7: 'ch7',8: 'ch8',9: 'ch9',10: 'ch10',11: 'ch11',12: 'ch12',13: 'ch13',14: 'ch14',15: 'ch15',16: 'ch16',17: 'ch17',18: 'ch18',19: 'ch19', 20:'ch20',21: 'ch21',22: 'ch22',23: 'ch23',24: 'ch24',25: 'ch25',26: 'ch26',27: 'ch27',28: 'ch28',29: 'ch29',30: 'ch30',31: 'ch31',32: 'ch32',33: 'quad_Tmp',34:'Err',35:'pAp',36:'pRp',37:'Int'})
    skyopc.dropna(inplace=True)
    skyopc = skyopc.set_index('Date')
    skyopc = skyopc.sort_values('Date')
    skyopc.index = pd.DatetimeIndex(skyopc.index)
    skyopc = skyopc[~skyopc.index.duplicated()]
    # remove repeated channel 16
    del skyopc['ch16']

    # Units: counts/100ml == 100 counts/cm3
    # Calculate total counts/cm3 by adding bins
    skyopc['SKYOPC_conc']=skyopc['ch1']+skyopc['ch2']+skyopc['ch3']+skyopc['ch4']+skyopc['ch5']+skyopc['ch6']+skyopc['ch7']+skyopc['ch8']+skyopc['ch9']+skyopc['ch10']+skyopc['ch11']+skyopc['ch12']+skyopc['ch13']+skyopc['ch14']+skyopc['ch15']+skyopc['ch17']+skyopc['ch18']+skyopc['ch19']+skyopc['ch20']+skyopc['ch21']+skyopc['ch22']+skyopc['ch23']+skyopc['ch24']+skyopc['ch25']+skyopc['ch26']+skyopc['ch27']+skyopc['ch28']+skyopc['ch29']+skyopc['ch30']+skyopc['ch31']+skyopc['ch32']
    skyopc['SKYOPC_conc']=skyopc['SKYOPC_conc']/100 #counts/cm3
    skyopc['SKYOPC_conc']=skyopc['SKYOPC_conc'].astype(float)
    
    return skyopc

try:
    SKYOPC = get_skyopc(d_loc,yesterday,today)
except:
    print('No SKYOPC data')




#######################################################################################

# Function to read and process CLASP data
# Inputs

def get_clasp(d_loc,d1,d2,claspn,channels,calfile,sf):
    # Function to convery interger to binary.
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    os.chdir(d_loc+'Data/')                  # Change directory to where the data is
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
    param_dates = np.asarray(dates)[np.where(statusaddr==0)[0][0:param_len]]
    if len(param_dates)<param_len:
        param_len = len(param_dates)
    
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
    CLASP_df = pd.DataFrame({'Date':dates, 'Heater flag':heater,
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
    flow_correction = ((flow/60)*1000)/sf # Sample volume in ml/s

    # Interpolate flow correction onto full timeseries and add to array
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    nans, x= nan_helper(flow_correction)
    flow_correction[nans]= np.interp(x(nans), x(~nans), flow_correction[~nans])
    CLASP_df['Sample volume (ml/s)']=flow_correction

    # Now to plot concentrations in counts/ml, just need to divide the counts/s by the sample volume
    CLASP_df['total_counts']=CLASP_df[1].astype(float)+ CLASP_df[2].astype(float)+CLASP_df[3].astype(float)+CLASP_df[4].astype(float)+CLASP_df[5].astype(float)+CLASP_df[6].astype(float)+CLASP_df[7].astype(float)+CLASP_df[8].astype(float)+CLASP_df[9].astype(float)+CLASP_df[10].astype(float)+CLASP_df[11].astype(float)+CLASP_df[12].astype(float)+CLASP_df[13].astype(float)+CLASP_df[14].astype(float)+CLASP_df[15].astype(float)+CLASP_df[16].astype(float)
    # CLASP-G flowrate = 3L/minute = 50 cm3/second
    # Units: particle counts/ sample interval
    # Sample interval: 1s
    # Calculate total counts
    # Calculate concentation
    
    CLASP_df['CLASP_conc'] = CLASP_df['total_counts'] / 50 #counts/cm3
    
    return CLASP_df


channels = 16 # Number of aerosol concentration channels (usually 16)
try:
    CLASP = get_clasp(d_loc,yesterday,today,'CLASP_F',16,calfile,1)
except:
    print('No CLASP data')

###############################################################################

# Set up y lim
max_counts = max(cpc_count)
if max_counts<100:
    yulim = 100
else:
    yulim = max_counts+10

# Plot & save


fig = plt.figure(figsize=(17,4))
ax = fig.add_subplot(111)
ax.grid(True)
try:
    ax.semilogy(cpc_dates,cpc_count, label='CPC (>5nm)',zorder=3,alpha=0.8)
except:
    pass

try:
    ax.semilogy(MSF_OPC.index,MSF_OPC.total_counts,label = 'MSF OPC (0.38-17$\mu$m)',zorder=4)  
except:
    pass

try:
    ax.semilogy(TAWO_OPC.index,TAWO_OPC.total_counts,label = 'TAWO OPC (0.38-17$\mu$m)',zorder=2)
except:
    pass

try:
    ax.semilogy(SKYOPC.index,SKYOPC.SKYOPC_conc,label = 'Sky OPC (0.25-32$\mu$m)',zorder=3)
except:
    pass

try:
    ax.semilogy(CLASP.index,CLASP.total_counts,label = 'CLASP',zorder=1)
except:
    print('failed at semilogy CLASP')
    pass
    
ax.set_ylim(0,1000)
ax.set_ylabel('Total Particle Counts / cm$^3$')
ax.set_title('Total Particle Counts: %s'%((dt.datetime.strftime(yesterday,'%Y-%m-%d')+' to '+dt.datetime.strftime(today,'%Y-%m-%d'))))
ax.set_xlabel('Hours (UTC)')
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(rule)
ax.set_xlim(yesterday,today)
ax.legend(loc='best',fontsize=10)
fig.tight_layout()
fig.savefig(d_loc + 'Ncounts_current.png')

fig.clf()


###############################################################################
# Plot spectra
###############################################################################

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
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    for i in range(0,len(dists)):
        if not dists[i]:
            continue
        else:
            ax.loglog(dists[i][0],dists[i][1],label=labels[i])

    ax.set_xlim(xlims[0],xlims[1])
    ax.set_xticks([0.35, 0.46, 0.66, 1, 1.7, 3,6.5,10, 16, 25, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Diameter (d) / $\mu$m')
    ax.set_ylabel('dN/dlogd (cc$^{-3}$)')
    ax.legend(loc='best',fontsize=10)
    ax.set_title('Aerosol size distribution: %s'%((dt.datetime.strftime(yesterday,'%Y-%m-%d')+' to '+dt.datetime.strftime(today,'%Y-%m-%d'))))
    fig.tight_layout()
    fig.savefig(d_loc + 'Spectra_current.png')
    fig.clf()
    # return fig or save fig.
    
def plot_dist_time(d1,d2,dates,counts,bins,name,vmax):
    fig = plt.figure(figsize=(17,5))
    ax = fig.add_subplot(111)
    cs = plt.pcolormesh(dates,np.arange(0,len(bins)-1,1),np.transpose(counts),vmin=0,vmax=vmax)
    cb = plt.colorbar(cs,extend='max',label='Counts/ cc',orientation='horizontal',pad=0.18,aspect=50,shrink=0.7)
    ax.xaxis_date()
    ax.set_title('%s: %s'%(name,(dt.datetime.strftime(d1,'%Y-%m-%d')+' to '+dt.datetime.strftime(d2,'%Y-%m-%d'))))
    ax.set_yticks(np.arange(0,len(bins)-1,1))
    ax.tick_params(axis='y', which='major', labelsize=10) 
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_yticklabels(bins)
    ax.set_ylabel('Particle Size ($\mu$m)')
    ax.set_xlabel('Hours UTC')
    ax.set_xlim(d1,d2)
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(rule)   
    fig.tight_layout()
    fig.savefig(d_loc + '%s_timeseries_current.png'%name)
    fig.clf()
    
# Subset counts.
try:
    MSF_counts = MSF_OPC[MSF_OPC.columns[0:24]]
    MSF_counts = MSF_counts.apply(pd.to_numeric, errors='coerce')
except:
    pass

try:
    TAWO_counts = TAWO_OPC[TAWO_OPC.columns[0:24]]
    TAWO_counts = TAWO_counts.apply(pd.to_numeric, errors='coerce')
except:
    pass
try:
    SKYOPC_counts = SKYOPC[SKYOPC.columns[0:31]]
    SKYOPC_counts = SKYOPC_counts.apply(pd.to_numeric, errors='coerce')
except:
    pass
try:
    CLASP_counts = CLASP[CLASP.columns[1:17]]
    CLASP_counts = CLASP_counts.apply(pd.to_numeric, errors='coerce')
except:
    print('failed at clasp counts conversion')
    pass

OPC_bins = 24
OPC_bounds = [0.35, 0.46, 0.66, 1, 1.3, 1.7, 2.3, 3, 4, 5.2, 6.5, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 37, 40]
SKYOPC_bins = 31
SKYOPC_bounds = [0.25,0.28,0.3,0.35,0.4,0.45,0.5,0.58,0.65,0.7,0.8,1.0,1.3,1.6,2,2.5,3,3.5,4,5,6.5,7.5,8.5,10,12.5,15,17.5,20,25,30,32,40]
CLASP_bins=16
CLASP_bounds=[0.3,0.4,0.5,0.6,0.7,1,2,3,4,5,6,7,8,9,10,12,14.3] # Note clasp bounds are radii
CLASP_bounds=list(np.asarray(CLASP_bounds)/2)


try:
    MSF_dist = get_dist(MSF_counts,OPC_bins,OPC_bounds)
except:
    MSF_dist=[]
try:
    TAWO_dist = get_dist(TAWO_counts,OPC_bins,OPC_bounds)
except:
    TAWO_dist=[]
try:    
    SKYOPC_dist= get_dist(SKYOPC_counts,SKYOPC_bins,SKYOPC_bounds)
except:
    SKYOPC_dist=[]
try:    
    CLASP_dist= get_dist(CLASP_counts,CLASP_bins,CLASP_bounds)
except:
    CLASP_dist=[]

plot_dist([MSF_dist,TAWO_dist,SKYOPC_dist,CLASP_dist],['MSF_OPC','TAWO_OPC','SKYOPC','CLASP'],[SKYOPC_bounds[0],SKYOPC_bounds[-1]])






# Plot time series distributions: 
try:
    sky_counts = SKYOPC_counts.to_numpy()
    sky_dates = np.array(SKYOPC_counts.index.to_pydatetime())
    plot_dist_time(yesterday,today,sky_dates,sky_counts,SKYOPC_bounds,'SKYOPC',1000)
except:
    pass

try:
    opc_counts = MSF_counts.to_numpy()
    opc_dates = np.array(MSF_counts.index.to_pydatetime())
    plot_dist_time(yesterday,today,opc_dates,opc_counts,OPC_bounds,'MSF_OPC',150)
except:
    pass

try:
    opct_counts = TAWO_counts.to_numpy()
    opct_dates = np.array(TAWO_counts.index.to_pydatetime())
    plot_dist_time(yesterday,today,opct_dates,opct_counts,OPC_bounds,'TAWO_OPC',150)
except:
    pass

try:
    clasp_counts = CLASP_counts.to_numpy()
    clasp_dates = np.array(CLASP_counts.index.to_pydatetime())
    plot_dist_time(yesterday,today,clasp_dates,clasp_counts,CLASP_bounds,'CLASP_F',150)
except:
    pass

