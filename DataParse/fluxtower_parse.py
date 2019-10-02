
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
import io
import itertools
import datetime
import pandas as pd

# KT15 parsing function
def extract_KT_data(start,stop,dpath,logf):
    # Extract KT15 data into a pandas array
    # Data format: YYYY MM DD HH MM.mmm TT.tt C
    # TT.tt = temperature, C = celcius

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(logf,'w')             # Open the log file for writing
    all_files = glob.glob('*.KT15')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    KT = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty or contains non-ascii characters
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue

        # Filter and report files with non-ascii characters
        content = open(f).read()
        try:
            content.encode('ascii')
        except UnicodeDecodeError:
            log.write("Error with: %s contains non-ascii characters.\n"%f)  
            continue
        
        # Store good data      
        KT = KT.append(pd.read_csv(f, header=None, delim_whitespace=True))
       
    # Sort out the date referencing and columns
    if KT.empty==False:
        KT[5] = KT[5].astype(int)
        KT['Date'] = pd.to_datetime(KT[0]*10000000000+KT[1]*100000000+KT[2]*1000000+KT[3]*10000+KT[4]*100+KT[5],format='%Y%m%d%H%M%S')
        KT = KT.set_index('Date')
        del KT[0],KT[1],KT[2],KT[3],KT[4],KT[5]
        KT.columns = ['T', 'Units']
        KT = KT.sort_values('Date')
        new_idx = pd.date_range(pd.to_datetime(str(start_f),format='%y%m%d'),pd.to_datetime(str(stop_f),format='%y%m%d'),freq='1s' )
        KT.index = pd.DatetimeIndex(KT.index)
        KT = KT[~KT.index.duplicated()]
        KT= KT.reindex(new_idx, fill_value=np.NaN)
        log.write('Data parse finished\n')
    else:
        log.write('No KT data found for this period')
    
    log.close()
    return KT

# SnD parsing

def extract_snd_data(start,stop,dpath,log_snd):
#aa;D.DDD;QQQ; VVVVV;CC
#aa = 33 (serial address of sensor)
#D.DDD = Distance to target in m (will need temperature adjustment
#QQQ = Data quality, varies beteen 152-600, 600 is the poorest quality
#VVVVV = diagnostic tests (only first two are actually something), 1 = pass. 
#CC = two-character checksum of data packet. (indication of data errors? Not sure how to read this. )
    
    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_snd,'w')             # Open the log file for writing
    all_files = glob.glob('*.SnD')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    snd = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        dat_lines =open(f).readlines()
        good_lines = [y for y in dat_lines if len(y)==49]
        date =[]
        x=[]
        Q=[]
        V=[]
        C=[]
        for i in range(0,len(good_lines)):   
            date.append((pd.to_datetime(good_lines[i][0:23],format='%Y %m %d %H %M %S.%f')).round('min'))
            Q.append(good_lines[i][35:38])
            V.append(good_lines[i][39:41])
            C.append(good_lines[i][45:47])
            try: 
                x.append(float(good_lines[i][29:34]))
            except ValueError:
                x.append(np.nan)

        snd = snd.append(pd.DataFrame({'Date':date,'depth':x,'Q':Q,'V':V,'C':C}))
    
    if snd.empty==False:
        snd = snd.set_index('Date')
        snd = snd.sort_values('Date')
    #    new_idx = pd.date_range(pd.to_datetime(str(start_f),format='%y%m%d'),pd.to_datetime(str(stop_f),format='%y%m%d'),freq='min' )
        snd.index = pd.DatetimeIndex(snd.index)
        snd = snd[~snd.index.duplicated()]
    #   snd= snd.reindex(new_idx, fill_value=np.NaN)
        
        # Check diagnostic tests pass.
        snd = snd[snd['V'].astype(int)==11]

        # Check data quality
        snd['Q'] = snd['Q'].astype(int)
        snd = snd[snd['Q']>151]

        # Check for crazy values
        snd['diff']=snd['depth'].diff()
        snd = snd[np.abs(snd['diff'])<0.1] # Ignore if 10 minute change greater than 10cm in 10 minutes
        
    else: 
        log.write('No data from snd\n')
       
    log.write('Data parse finished\n')
    log.close()

    return snd

# HMP155 parsing

def HMP_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df = df.dropna()
        df['Second'] = df['Second'].astype(float)
        df['Second'] = df['Second'].astype(int)
        df['Minute'] = df['Minute'].astype(int)
        df['Hour'] = df['Hour'].astype(int)
        df['Day'] = df['Day'].astype(int)
        df['Month'] = df['Month'].astype(int)
        df['Year'] = df['Year'].astype(int)
        df['Date'] = pd.to_datetime(df['Year']*10000000000+df['Month']*100000000+df['Day']*1000000+df['Hour']*10000+df['Minute']*100+df['Second'],format='%Y%m%d%H%M%S')
        df = df.set_index('Date')
        del df['Year'],df['Month'],df['Day'],df['Hour'],df['Minute'],df['Second'],df['junk']
        df.columns = ['RH', 'Ta', 'Tw', 'Err', 'h']
        df['RH']=df['RH'].astype(float)
        df['Tw']=df['Tw'].astype(float)
        df['Ta']=df['Ta'].astype(float)
        #df['h']=df['h'].astype(int)       
        df = df.sort_values('Date')
        #new_idx = pd.date_range(pd.to_datetime(start).round('1s'),pd.to_datetime(stop).round('1s'),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        #df= df.reindex(new_idx, method='nearest')
    else:
        df = pd.DataFrame(columns=['RH', 'Ta', 'Tw', 'Err', 'h'])
    return df

def extract_HMP_data(name, start,stop,dpath,logf, save):
    # Extract HMP155 data into a pandas array
    # Data format: YYYY MM DD HH MM.mmm TT:TT:TT RH Ta Tw Err hs
    # Ta = seperate probe T, Tw = wetbulb t, hs = heating status

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(logf,'w')             # Open the log file for writing
    all_files = glob.glob('*.%s'%name)  # List all data files
    
    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    HMP = pd.DataFrame()

    # Extract the data
    for f in dfs:
        # Ignore file if it's empty 
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        fd = io.open(f,"r",errors='replace')
        f_dat = fd.readlines()
        clean_dat = [i for i in f_dat if len(i)>=60 and len(i)<=63]
        pdf = pd.DataFrame(clean_dat)
        pdf[1] = pdf[0].str.split()
        final_df = pd.DataFrame(pdf[1].values.tolist(), columns=['Year','Month','Day','Hour','Minute','Second','junk','RH','Ta', 'Tw', 'Err', 'h'])
        # Store good data
        HMP = HMP.append(final_df)
        
 
    # Sort out the date referencing and columns
    HMP = HMP_pdf_sort(HMP,start,stop)

    log.write('Data parse finished\n')
    if save: 
        HMP.to_csv(save+'%s_%s'%(name,str(start.date())))
        log.write('Saved csv')

    log.close()
    return HMP

# Ventus parsing

def ventus_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df[5] = df[5].astype(int)
        df['Date'] = pd.to_datetime(df[0]*10000000000+df[1]*100000000+df[2]*1000000+df[3]*10000+df[4]*100+df[5],format='%Y%m%d%H%M%S')
        df = df.set_index('Date')
        del df[0],df[1],df[2],df[3],df[4],df[5]
        df.columns = ['wsd', 'wdir', 'T', 'Checksum']
        df = df.sort_values('Date')
        new_idx = pd.date_range(pd.to_datetime(str(start),format='%y%m%d'),pd.to_datetime(str(stop),format='%y%m%d'),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        df= df.reindex(new_idx, fill_value=np.NaN)
    return df

def extract_ventus_data(start,stop,dpath,log_ventus):
    # Extract Ventus data into a pandas array
    # Data format:
    #<STX>SS.S DDD +TT.T xx*XX<CR><ETX>
    #        SS.S = wind speed (m/s)
    #        DDD = wind direction
    #        +TT.T = signed virtual temperature
    #        xx = status
    #        XX = checksum

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_ventus,'w')             # Open the log file for writing
    all_files = glob.glob('*.ventus*')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    v1 = pd.DataFrame()
    v2 = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty or contains non-ascii characters
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        # Filter and report files with non-ascii characters
        content = open(f).read()
        try:
            content.encode('ascii')
        except UnicodeDecodeError:
            log.write("Error with: %s contains non-ascii characters.\n"%f)  
            continue
            
        if f[-1]=='1':
            v1 = v1.append(pd.read_csv(f, header=None, delim_whitespace=True, error_bad_lines=False))
        if f[-1]=='2':
            v2 = v2.append(pd.read_csv(f, header=None, delim_whitespace=True, error_bad_lines=False))
        
    # Sort out the date referencing and columns
    v1 = ventus_pdf_sort(v1,start_f,stop_f)
    v2 = ventus_pdf_sort(v2,start_f,stop_f)
    log.write('Data parse finished\n')
    log.close()
    return v1,v2


# Metek parsing

def metek_pdf_sort(df,start,stop):
    # Sort out the date referencing and columns
    if df.empty==False:
        df['ms']= (df[5] - np.floor(df[5]))*1000000
        df['ms']= df['ms'].astype(int)
        df[0] = df[0].astype(str)
        df[1] = df[1].astype(str).apply(lambda x: x.zfill(2))
        df[2] = df[2].astype(str).apply(lambda x: x.zfill(2))
        df[3] = df[3].astype(str).apply(lambda x: x.zfill(2))
        df[4] = df[4].astype(str).apply(lambda x: x.zfill(2))
        df[5] = np.floor(df[5]).astype(int)
        df[5] = df[5].astype(str).apply(lambda x: x.zfill(2))
        df['ms'] = df['ms'].astype(str).apply(lambda x: x.zfill(6))   
        
        df['Date'] = pd.to_datetime(df[0]+df[1]+df[2]+df[3]+df[4]+df[5]+df['ms'],format='%Y%m%d%H%M%S%f')
        df = df.set_index('Date')
        del df[0],df[1],df[2],df[3],df[4],df[5],df['ms'],df[7],df[9],df[10],df[12],df[13],df[15],df[16]
        df.columns = ['Status','x','y','z','T']
        df = df[pd.to_numeric(df['T'], errors='coerce').notnull()]        
        df['T']=df['T'].astype(float)
        df['x']=df['x'].astype(float)
        df['y']=df['y'].astype(float)
        df['z']=df['z'].astype(float)
        df['T']=df['T']/100
        df = df.sort_values('Date')
        #new_idx = pd.date_range(pd.to_datetime(str(start),format='%y%m%d'),pd.to_datetime(str(stop),format='%y%m%d'),freq='1s' )
        df.index = pd.DatetimeIndex(df.index)
        df = df[~df.index.duplicated()]
        #df= df.reindex(new_idx, fill_value=np.NaN)
        # Change units from cm/s to m/s, (* 0.01), and T to kelvin
        df['x']=df['x']*0.01
        df['y']=df['y']*0.01
        df['z']=df['z']*0.01
        df['T']= df['T']+ 273.15
    return df

def extract_metek_data(start,stop,dpath,log_metek,save=False):
    # Extract metek data into a pandas array
    # Data format:
    #2019 04 02 16 23 41.734 M:x =    14 y =    -1 z =    12 t =  2357
    # M = measured data heater off
    # H = measured data heater on
    # D = measured data heater defect
    # x,y,z componenets of wind in cm/s
    # t = acoustic temperature 2357 = 23.57C
    
    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_metek,'w')             # Open the log file for writing
    all_files = glob.glob('*.metek*')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    m1 = pd.DataFrame()
    m2 = pd.DataFrame()

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue
        
        if f[-1]=='1':
            try:
                m1 = m1.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False))
            except:
                print('Data error with %s'%f)
                continue
        if f[-1]=='2':
            try:
                m2 = m2.append(pd.read_csv(f, header=None, delim_whitespace=True,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], error_bad_lines=False))
            except:
                print('Data error with %s'%f)
                continue
               
        
    # Sort out the date referencing and columns
    m1 = metek_pdf_sort(m1,start_f,stop_f)
    m2 = metek_pdf_sort(m2,start_f,stop_f)
    log.write('Data parse finished\n')
    
    # crop data for date/time
    m1=m1[start:stop]
    m2=m2[start:stop]
    
    if save: 
        m1.to_csv(save+'metek1_%s'%str(start.date()))
        m2.to_csv(save+'metek2_%s'%str(start.date()))
        log.write('Saved csv')
    log.close()
    
    return m1,m2     





# Licor parsing

def extract_licor_data(start,stop,dpath,log_licor,save=False):

#2019 04 03 11 11 56.453 89	189	0.16469	35.4518	0.04404	297.105	20.74	99.0	1.5224
# Date, Ndx, DiagVal, CO2R, CO2D, H2OR, H2OD, T, P, cooler

    os.chdir(dpath)                  # Change directory to where the data is
    log = open(log_licor,'w')             # Open the log file for writing
    all_files = glob.glob('*.licor')  # List all data files

    # Get start and stop filenames
    start_f = int(start.strftime("%y%m%d"))
    stop_f = int(stop.strftime("%y%m%d"))

    # Extract daterange
    file_dates = np.asarray([int(f[0:6]) for f in all_files])
    idxs = np.where(np.logical_and(file_dates>=start_f, file_dates<=stop_f))[0]
    dfs = [all_files[i] for i in idxs]

    # Initialise empty data frames
    licor = pd.DataFrame()

    # Function to convert to float otherwise write nan
    def convert_float(x):
        try:
            return np.float(x)
        except:
            return np.nan

    # Extract the data
    for f in dfs: 
        # Ignore file if it's empty
        if os.path.getsize(f)==0:
            log.write('Error with: '+f+' this file is empty.\n')
            continue 
        licor = licor.append(pd.read_csv(f, header=None, sep='\t',error_bad_lines=False))
    
    # Sort out the date referencing and columns
    if licor.empty==False:
        licor['Date']=licor[0].str[0:23]
        licor = licor[licor['Date'].map(len) ==23]
        licor['Date'] = pd.to_datetime(licor['Date'],format='%Y %m %d %H %M %S.%f')
        licor['Ndx']=licor[0].str[24:].astype('int')
        licor['DiagV']=licor[1].astype('int')
        licor['CO2R'] = licor[2]
        licor['CO2R']=licor['CO2R'].apply(convert_float)
        licor['CO2D'] = licor[3]
        licor['CO2D']=licor['CO2D'].apply(convert_float)/1000 # mol/m3
        licor['H2OR'] = licor[4]
        licor['H2OR']=licor['H2OR'].apply(convert_float)
        licor['H2OD'] = licor[5]
        licor['H2OD']=licor['H2OD'].apply(convert_float)/1000 # mol/m3
        licor['T'] = licor[6].astype('float')+273.15      # K
        licor['P'] = licor[7].astype('float')*1000        # Pa
        licor['cooler'] = licor[8].astype('float')
        del licor[0],licor[1],licor[2],licor[3],licor[4],licor[5],licor[6],licor[7],licor[8],licor[9]                 
        licor = licor.sort_values('Date')
        licor = licor.set_index('Date')
    else:
        log.write('No data from licor\n')
        
        # crop data for date/time
    licor=licor[start:stop]        
    log.write('Data parse finished\n')
    if save: 
        licor.to_csv(save+'licor_%s'%str(start.date()))
        log.write('Saved csv')
        
    log.close()

    return licor    
