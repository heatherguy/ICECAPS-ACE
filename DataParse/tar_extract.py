#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 12:06:02 2020

@author: heather
"""

from utils import *
import numpy as np       
import datetime as dt  


# Data names: 
# HMP1: 2m T/RH
# HMP2: 4m T/RH
# HMP3: 8m T/RH
# HMP4: 15m T/RH
# metek1: 2m 3D sonic
# metek2: 15m 3D sonic
# licor: H2O and CO2 analyzer
# KT15: Snow surface temp
# ventus1: 4m 2D sonic
# ventus2: 8m 2D sonic
# SnD: Snow depth sensor.

# What data do you want to get?
#dname = 'SKYOPC'
#dname = 'Summit_MSF_ICECAPSACE_OPCN3'
dname = 'KT'

#7,6,5,

# Where are the raw .tar files?
#dloc = '/Users/heather/ICECAPS-ACE/Data/raw/'
#dloc = '/Volumes/Data/ICECAPSarchive/ace/raw/'
dloc = '/Volumes/Data/ICECAPSarchive/fluxtower/raw/'

# Where do you want to store the extracted .tar files? 
#extract_out = '/Users/heather/ICECAPS-ACE/temp_data/'
#extract_out = '/Volumes/Data/ICECAPSarchive/fluxtower/raw_extracted/Metek/'
#extract_out = '/Volumes/Data/ICECAPSarchive/ace/Extracted/CLASP/'
extract_out = '/Volumes/Data/ICECAPSarchive/fluxtower/raw_extracted/KT/'

# Extract all tar files. 
extract_tar(dloc,extract_out,dname)