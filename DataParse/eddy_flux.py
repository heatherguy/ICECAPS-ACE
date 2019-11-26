#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:41:17 2019

@author: heather
"""

import numpy as np      
import datetime as dt
from scipy.signal import medfilt, detrend, coherence, windows
import pandas as pd


# Contants

# For air denisty and heat capacity, for now lets use a constant assuming dry air and temps of -15C, pressure of 673mb.
# The licor measures air pressure, can probably make a better estimate using licor data (rho_dry = P/RT)
# Can probably also use temperature and humidty values to make a better estimate of heat capacity

rho = 0.8136 # kg/m3
cp = 1003    # j/kg*K
Ma = 28.96    # Molar mass of dry air (g/mol)
Mh = 18.01528 # Molar mass of H2O (g/mol)
R = 8.314     # J/K/mol #universal gas constant
# L = latent heat of vaporisation of water = 2264705 j/kg
L = 2264705.0 # j/kg


def replace_outliers(var,sd):
    # replace outliers with median filter
    var=var.astype(float)
    jj = ~np.isnan(var) # Ignore nans
    temp = var[jj]
    mf = medfilt(temp,11)             # Get median filter
    ii = np.abs(temp - mf) > 3*sd     # Get outliers, where greated than 3 Sd's from median
    temp[ii] = mf[ii]                 # Replace these outliers with the median
    var_clean = var
    var_clean[jj] = temp      # Put back into orginal array/
    
    return var_clean


def clean_metek(m_in):
# Clean Metek data
# Based on clean_metek.m by IAM, 25/7/2014
# metek = clean_metek(metek)
# INPUT
#  metek  - data strucure
# OUTPUT
#  metek  - data structure
# wind compents and sonic temperature are cleaned up:
# - the time series are filtered for outliers, these are replaced with  
#   median filtered values
# - missing values from error messages (NaNs) are then interpolated over
    
 # filter for clear outliers - replace with median filtered values
#set limit at 3*standard deviation of vertical wind component
    m_out = m_in
    jj = ~np.isnan(m_in.z) # Not nan indices
    m1_sd = np.std(m_in.z[jj]) # standard deviation of vertical wind component. 
    m_out['x']=replace_outliers(m_in['x'],m1_sd)
    m_out['y']=replace_outliers(m_in['y'],m1_sd)
    m_out['z']=replace_outliers(m_in['z'],m1_sd)
    m_out['T']=replace_outliers(m_in['T'],m1_sd)

# patch up missing data from records with error messages by interpolation
# metek output should be 10Hz
# 10Hz = 0.1s = 100 ms
    m_out = m_out.resample('100L').mean().interpolate()
    return m_out


def Ts_sidewind_correction(Ts,u,v,w):
# Do cross wind temperature correction
# Adapted from Ts_sidewind_correction.m by IMB July 2007
#     Correct contamination of sonic temperature measurement for lengthening of
#     sonic path by sidewind. 
#
#function T = Ts_sidewind_correction(Ts,u,v,w,model);
# INPUTS
#  Ts    : sonic temperature (K)
#  u,v,w : wind components in sonic measurement frame (before any rotations,
#          motion correction, etc) (m/s)

# OUTPUT
#  T     : corrected temperature (K)
#
#     Correction follows van Dijk et al. 2004: The
#     principles of surface flux physics: theory, practice and description of
#     the ECPACK library (www.met.wau.nl/projects/jep). See also: Liu et al.
#     2001: BLM, 100, 459-468, and Schotanus et al. 1983: BLM, 26, 81-93.
    vn2 = (3/4)*(u**2 + v**2) + 0.5*w**2
    T = Ts + vn2/403
    return T



def rotate_to_run(m,avp):
# Correct tilt and align with horizontal streamlinge over a single run (~20mins)
# Adapted from rotate_to_run.m by IMB July 2006
#% references:
#%  Wilczak et al. 2001: sonic anemometer tilt corrections. BLM, 99, 127-150
#%  (but beware typos in equations)
#%  Kaimal & Finnigan, 1994: Atmospheric Boundary Layer Flows: Their
#%  Structure and Measurement. Oxford University Press
#%  van Dijk et al. 2004: The principles of surface flux physics: theory,
#%  practice and description of the ECPACK library
#%  www.met.wau.nl/projects/jep
    #% First rotate to align x-axis with mean wind direction in sonic's
    #% reference frame
    
    m_g = m.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    m_out = pd.DataFrame(columns=['x','y','z','T','u','v','w','theta','phi'])
    for group in m_g:
        m = group[1]
        theta=np.arctan2(np.mean(m['y']),np.mean(m['x']))
        u1 = m['x']*np.cos(theta) + m['y']*np.sin(theta)
        v1 = -m['x']*np.sin(theta) + m['y']*np.cos(theta)
        w1 = m['z']

        #% Next rotate u and w so that x-axis lies along mean streamline and 
        #% mean(w) is zero
        phi = np.arctan2(np.mean(w1),np.mean(u1))
        m['u'] = u1*np.cos(phi) + w1*np.sin(phi)
        m['v'] = v1
        m['w']=  -u1*np.sin(phi) + w1*np.cos(phi)

        # Theta is angle of rotation um-to-vm (anticlockwise or righthanded)
        # to aign u with mean wind (degrees)
        m['theta'] = theta*180/np.pi

        # phi is tilt angle (+ve tilts x-axis upwards) to align x-axis with
        # mean streamline and force <w>=0
        m['phi'] = phi*180/np.pi
        m_out = m_out.append(m)
        
        # Output wind components in streamline oriented reference frame, theta and phi
    return m_out



def eddyflux(x,y):
# Function to calculate instantaneous flux 
    # x is time series of vertical velocity
    # y is parameter of interest.
    x = detrend(x)
    y = detrend(y)
    flux = np.mean(x*y) # Instantaneous flux
    std = np.std(x*y) # Standard deviation of instantaneous flux
    return flux,std



def ogive(x, y, sf,avp):
# Calculate cospectral density and ogive function.    
# Choose your two variables. 
#x = m1['w']
#y = m1['u']
#sf = 10.0 # Sampling frequency    
    f_list=[]
    Csdxy_list=[]
    ogive_list=[]
    x_g = x.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    y_g = y.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    keys = list(x_g.groups.keys())
    
    for i in range(0,len(keys)):
        x =x_g.get_group(keys[i])
        y =y_g.get_group(keys[i])
        m = len(x)  
        # Make sure m is even
        m=np.fix(m/2)*2
        df = sf/m   # Delta frequency
        f = np.arange(df,sf/2+df,df) # Frequency series

        # Calculated using mathematical methods, fast fourier transform
        # FFT gives the amplitudes of variation for different frequencies
        # Calculate the FFT for x and for y. (normalised by the length of the time series)
        xw=detrend(x)
        yw=detrend(y)
        Rxx = np.fft.fft(xw)/m
        Ryy = np.fft.fft(yw)/m

        #The cospectrum is:
        Sxy = Rxx*np.conj(Ryy)

        #The fourier spectra are complex and symmetrical about a frequency of 0
        #(ie they have fourier components for both +ve and -ve frequency) -
        #negative frequencies don't make sense in the real world, so we 'fold'
        #the spectra about 0 and store as Gxy etc

        Gxy = 2 * Sxy[1:int(m)/2]
        Gxy =np.append(Gxy, Sxy[int(m)/2+1])

        # Co-spectra: how much different sized eddies contribute to the covariance of two variables - the product of the fft of two variables. 
        #Gxx, Gyy are the power spectra of x, y, Gxy is the cospectrum of x and
        #y. The (co)spectral densities are simply these divided by the frequency
        #interval to give power per unit frequency. The integral of this is the
        #variance (or covariance) of the time series.

        Csdxy = Gxy / df

        # Ogive = cumulative co-spectrum. Use to find the point after which there is no flux contribution. Averaging peroiod should be longer
        # than this so large eddies aren't lost, but otherwise as short as possible so that large scale changes (atmospheric) aren't included. 
        n = len(Csdxy)
        ogive = np.zeros(n)
        ogive[0]=(Csdxy*df)[0]
        for i in range(1,n):
            k=n-1-i
            ogive[k]=ogive[k+1]+(Csdxy*df)[k]
            
        f_list.append(f)
        Csdxy_list.append(Csdxy)
        ogive_list.append(ogive)
        
    return f_list,Csdxy_list,ogive_list



def shf(w,t,avp):
# Calculate sensible heat flux. 

# The EC method (e.g., Oke, 1987) calculates the covariance
# between the anomalies in the vertical wind (w') and temperature (T')
# to determine the turbulent sensible heat flux according to the following equation:
# SH = rho * cp * mean(w'*T')
# rho = density of air       
# cp = heat capacity of air

# To calculate mean(w'*T'), use ians eddyflux.m script
# Timeseries are first linearly detrended, then the mean covariance = flux is
# found 

# It represents the loss of energy by the
# surface by heat transfer to the atmosphere. It is positive when directed
# away from the surface into the atmosphere.
    SHF=[]
    stds=[]
    w_g = w.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    t_g = t.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    keys = list(w_g.groups.keys())
    
    for i in range(0,len(keys)):
        w1 =w_g.get_group(keys[i])
        t1 =t_g.get_group(keys[i])
        # Check there's at least 60% of averaging period data
        if len(t1) >= 0.6 * (15*60) * 10:
            flux,std = eddyflux(w1,t1) # Flux units = Km/s
            SHF.append(rho * cp * flux) # W/m2
            stds.append(std)
        else:
            SHF.append(np.nan)
            stds.append(np.nan)
    
    return SHF,std

def lhf(w,T,P,Nconc,avp):
    LHF=[]
    w_g = w.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    T_g = T.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    P_g = P.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    Nconc_g = Nconc.groupby(pd.Grouper(freq='%sMin'%avp)) # Split df into avp groups. 
    keys = list(w_g.groups.keys())
    
    for i in range(0,len(keys)):
        w1 =w_g.get_group(keys[i])
        try: 
            T1 = T_g.get_group(keys[i])
            P1 = P_g.get_group(keys[i])
            Nconc1 = Nconc_g.get_group(keys[i])
        except:
            #print('No licor data for %s'%str(keys[i]))
            LHF.append(np.nan)
            continue

        mass_conc_air = (Ma*P1)/(R*T1) / 1000       # kg air/ m3
        mass_conc_h2o = (Mh * Nconc1) / 1000   # kg water / m3
        q = mass_conc_h2o / mass_conc_air         # Mass mixing ratio

        # Average q onto the same time series as w.
        q = q.resample('100L').mean().interpolate(method='time',limit=3,limit_direction='both')
        
        # Join data frames so they're the same length, remove nans
        join = pd.concat([q, w1], axis=1, sort=False)
        join = join.dropna()
        
        # Check there's at least 60% of averaging period data
        if len(join) >= 0.6 * (15*60) * 10:
            # Calculate latent heat flux.
            #It represents a loss of energy from the
            #surface due to evaporation. 
            # LH = L * rho * mean(w'*q')
            # q = H2O mixing ratio
            flux,std = eddyflux(join['w'],join[0])
            LHF.append(L * rho * flux)
            #print('Good licor data')
        else:
            #print('Not enough good data for %s'%str(keys[i]))
            LHF.append(np.nan)
        
    return LHF