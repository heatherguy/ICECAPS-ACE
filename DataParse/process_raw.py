#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:08:16 2020

@author: heather
"""

from fluxtower_parse import *
from ace_parse import *
import numpy as np      
import datetime as dt



in_loc = '/Volumes/Data/ICECAPSarchive/fluxtower/raw_extracted/KT/'
out_loc = '/Volumes/Data/ICECAPSarchive/fluxtower/processed/KT/'

#in_loc = '/Volumes/Data/ICECAPSarchive/ace/Extracted/CLASP/'
#out_loc = '/Volumes/Data/ICECAPSarchive/ace/processed/CLASP/'


# For clasp:
claspn = 'CLASP_F'
#claspn = 'CLASP_F'
calfile = '/Volumes/Data/ICECAPSarchive/qc_files/calibration-unit-F-Feb2019.mat'

KT_qcf = '/Users/heather/ICECAPS-ACE/qc_files/KT_bad_dates'


# Extract and process all data, and save to 'processed'
# Start and stop date:
all_start = dt.datetime(2019,5,27,0,0)
all_stop = dt.datetime(2019,12,10,0,0)

all_days = pd.date_range(all_start,all_stop,freq='1D')


# Loop through and splite into daily files
for i in range(0,len(all_days)-1):
#i=0
    start = all_days[i]
    stop = all_days[i+1] - pd.Timedelta(seconds=0.1)
    print(str(start) + ' to ' + str(stop))
    
    extract_KT_data(start,stop,in_loc,KT_qcf,save=out_loc)
    
    #ventus = extract_ventus_data(start,stop,in_loc,logf,save=out_loc)
    #m1,m2 = extract_metek_data(start,stop,in_loc,save=out_loc)
    #licor = extract_licor_data(start,stop,in_loc,save=out_loc)
    #m1,m2 = extract_metek_data(start,stop,in_loc,save=out_loc)
    
    #extract_cpc(start,stop,in_loc,save=out_loc)
    #extract_skyopc(start,stop,in_loc,save=out_loc)
    #extract_opc('TAWO',start,stop,in_loc,save=out_loc)
    
    #HMP2 = extract_HMP_data('HMP2',start,stop,in_loc,logf,save=out_loc)
    #HMP3 = extract_HMP_data('HMP3',start,stop,in_loc,logf,save=out_loc)
    #HMP4 = extract_HMP_data('HMP4',start,stop,in_loc,logf,save=out_loc)
    #    print('Failed to process: %s'%str(start))

    #try:
    #    counts,params = get_skyopc(in_loc,start,stop,save=out_loc)
    #except:
    #    print('Parse failed!')
    
    # For clasp:
    #get_clasp(in_loc,start,stop,claspn,calfile,save=out_loc)