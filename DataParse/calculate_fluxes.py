#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:10:21 2020

@author: heather
"""

import numpy as np      
import datetime as dt
from scipy.signal import medfilt, detrend, coherence, windows
import pandas as pd
import os
import glob
from flux_functions import *


####### INPUTS #######
# Data location:
d_loc = '/Volumes/Data/ICECAPSarchive/fluxtower/processed/'

# Start and stop date:
start = dt.datetime(2019,11,21,0,0)
stop = dt.datetime(2019,11,27,0,0)

avp = 15  # Averaging time in minutes. 
Z = 1.2   # Measurement altitude in m
sf = 10   # sampling frequency (10Hz)


#############################################################

# Set up some things

# Days to loop through
days = pd.date_range(start,stop,freq='1D')
m = avp * 60 * 10 # Sample length of interval
# Make sure m is even
m=np.fix(m/2)*2
df = sf/m                    # Frequency intervals
f = np.arange(df,sf/2+df,df) # Frequency timeseries



# Loop through each day: 

# 2) Loop through each day

for day in days:
    day_str = str(day.date()) 
    print(day_str)
    
# 3) Initialize data frame

    # QC structure  0 = BAD, 1 = GOOD, 2 = OK, 3 = highly suspect
    QC_out = pd.DataFrame(columns=['Dates', 'skew_u', 'skew_v' ,'skew_w' ,'skew_T' ,'skew_q' ,'kurt_u' ,'kurt_v' ,'kurt_w' ,'kurt_T' ,'kurt_q' ,'sst_wu' ,'sst_wv' ,'sst_wt' ,'sst_wq' ,'Obukhov','BowenRatio','FrictionU','ZoverL','sigma_w','itc_w','quality_wu','quality_wv','quality_wt','quality_wq'])
    QC_out['Dates']= pd.date_range(day, day+pd.Timedelta(hours=24),freq='%s min'%avp)[0:-1]
    QC_out = QC_out.set_index('Dates')

    # Flux structure
    flux_out = pd.DataFrame(columns=['Dates', 'SHF', 'LHF','MOMFU','ubar','QC_LHF','QC_SHF','QC_MOMFU'])
    flux_out['Dates']= pd.date_range(day, day+pd.Timedelta(hours=24),freq='%s min'%avp)[0:-1]
    flux_out = flux_out.set_index('Dates')  
    
    # Co-spectra structure

    cospec_out = pd.DataFrame(columns=['Dates']+list(f))
    cospec_out['Dates']= pd.date_range(day, day+pd.Timedelta(hours=24),freq='%s min'%avp)[0:-1]
    cospec_out = cospec_out.set_index('Dates')  
    
    ogive_out = pd.DataFrame(columns=['Dates']+list(f))
    ogive_out['Dates']= pd.date_range(day, day+pd.Timedelta(hours=24),freq='%s min'%avp)[0:-1]
    ogive_out = ogive_out.set_index('Dates') 

# 4) Get the 3D sonic data

    if os.path.isfile(d_loc+'metek/metek1_%s'%day_str):
        m1_orig = pd.read_csv(d_loc+'metek/metek1_%s'%day_str, index_col=0, parse_dates=[0])
        if m1_orig.empty:
            print('Error: File empty, '+day_str)
            continue
    else:
        print('Error: File empty, '+day_str)
        continue
        
# 5) Get licor data

    if os.path.isfile(d_loc+'LiCOR/licor_%s'%day_str):
        licor = pd.read_csv(d_loc+'LiCOR/licor_%s'%day_str, index_col=0, parse_dates=[0])
        # QC licor data
        licor[licor['QC']!=1]=np.nan
    else:
        print('Error: File empty, '+day_str)
        continue
        
# 6) Get HMP 2m T data

    if os.path.isfile(d_loc+'HMP/HMP1_%s'%day_str):
        HMP1 = pd.read_csv(d_loc+'HMP/HMP1_%s'%day_str, index_col=0, parse_dates=[0])
        # Crop to date, time
        HMP1 = HMP1[day:day+pd.Timedelta(hours=24)] 
    else:
        print('Error: File empty, '+day_str)
        continue
        
# 7) Clean metek data 

    m1 = clean_metek(m1_orig)

# 8) Implement cross-wind temperature correction

    m1['T'] = Ts_sidewind_correction(m1['T'],m1['x'],m1['y'],m1['z'])

# 9) Rotate to average streamline for each averaging period. 

    m_rot = rotate_to_run(m1,avp)
    
# 10) Mask flow distortion from tower or camp (wind direction between 17 and 90 degrees) ? - this gets filtered out by the statistical testing anyway. 

    m_rot['wdir_arb'] = get_windd(m_rot['x'],m_rot['y']) # wind direction in sonic coords
    m_rot['wdir_cor'] = deg_rot(m_rot['wdir_arb'],-45)   # wind direction corrected for sonic orientation.
    
# FLOW DISTORTION MASK CURRENTLY NOT IMPLEMENTED!    
#    distortion_mask = m_rot['wdir_cor'].between(17, 100, inclusive = True)
#    m_rot['w'][distortion_mask] = np.nan                 # Mask vertical wind fluctuations where flow may be distorted. 
    
    
# 11) Calculate mass mixing ratio and wet air pp from licor 

    Ta = HMP1['Ta'] + 273.15   # 2m air temperature,  K
    P = licor['P']             # 2m air pressure form licor, Pa
    Nconc = licor['H2OD']      # H2O number concentration from licor, mol/m3
    m_rot['q'],m_rot['PPw'],m_rot['PPd'] = licor_mmr(Ta,P,Nconc)  # H2O mass mixing ratio, kg/kg
    
# 12) Estimate air density, absolute humidity, Cp and Lv  
    
    m_rot['rho'] = rho_m(Ta,P,m_rot['q'])        # Air density estimation, kg/m3
    m_rot['rho'] = m_rot['rho'].fillna(np.mean(m_rot['rho'][~np.isnan(m_rot['rho'])]))    

    # Calculate absolute humidity (A, kg K / J)
    # T - water and side wind corrected sonic temperature (K)
    # ppwet: water vapor partial pressure (pa
    # C = 2.16679; % g K J-1
    # A = (C * ppwet / T)/1000 # kg K / j
    # Fill empty values of A with mean value of A. 

    m_rot['A'] = ((2.16679 * m_rot['PPw']) / m_rot['T'])/1000.0 # kg K /j
    m_rot['A'] = m_rot['A'].fillna(np.mean(m_rot['A'][~np.isnan(m_rot['A'])]))

    # Calculate heat capacity CP
    # cp = 1000 * (1.005 + (1.82*A))  # Cp adjusted for absolute humidity in J/kg/K

    m_rot['Cp'] = 1000.0 * (1.005 + (1.82*m_rot['A']))  # Cp adjusted for absolute humidity in J/kg/K

    # Calculate latent heat of vaporisation
    # Lv = 3147.5 - (2.372 * Ta) # Lv adjusted for temperature (j/kg)

    m_rot['Lv'] = (3147.5 -( 2.73 * m_rot['T'])) * 1000
    
# 13) Split into runs based on averaging time

    m_g = m_rot.groupby(pd.Grouper(freq='%sMin'%avp))    
    keys = list(m_g.groups.keys())
    
    for i in range(0,len(keys)):
        # Initialise fluxes to run
        run_SHF = True
        run_LHF = True
        run_MOMFU = True
        
        k=keys[i]
        try:
            m1 = m_g.get_group(k)
        except:
            # If this part of the file is missing, skip all. 
            continue

# 14) Interpolate over data gaps that are smaller than around 5 minutes (60% data for 15 min period)

        m1 = m1.interpolate(limit=3000,limit_direction='both')
    
        if len(m1['w'][np.isnan])!=0:
            run_SHF = False
            run_LHF = False
            run_MOMFU = False
        elif len(m1['q'][np.isnan])!=0:
            run_LHF=False
        elif len(m1['T'][np.isnan])!=0:
            run_SHF=False
        elif len(m1['u'][np.isnan])!=0:
            run_MOMFU=False
            
# Save ubar for run
 
        flux_out.loc[k]['ubar'] = np.mean(m1['u'])
            
# 15) Calculate skew and kurtosis
    
        QC_out.loc[k]['skew_w'] = skew(m1['w'])
        QC_out.loc[k]['skew_u'] = skew(m1['u'])
        QC_out.loc[k]['skew_v'] = skew(m1['v'])
        QC_out.loc[k]['skew_T'] = skew(m1['T'])
        QC_out.loc[k]['skew_q'] = skew(m1['q'])

        QC_out.loc[k]['kurt_w'] = kurtosis(m1['w'])
        QC_out.loc[k]['kurt_u'] = kurtosis(m1['u'])
        QC_out.loc[k]['kurt_v'] = kurtosis(m1['v'])
        QC_out.loc[k]['kurt_T'] = kurtosis(m1['T'])
        QC_out.loc[k]['kurt_q'] = kurtosis(m1['q'])
        
    
# 16) Do stationarity testing for wu and wt

        if run_MOMFU:
            QC_out.loc[k]['sst_wu'], Cwu, rol_cov_wu = stationarity(m1['w'],m1['u'])
            QC_out.loc[k]['sst_wv'], Cwv, rol_cov_wv = stationarity(m1['w'],m1['v'])
            QC_out.loc[k]['sst_wt'], Cwt, rol_cov_wt = stationarity(m1['w'],m1['T'])
        
# 17) Calculate friction velocity

            QC_out.loc[k]['FrictionU'] = (Cwu**2 + Cwv**2)**(1/4)

# 18) Calculate obukhov length
    
            QC_out.loc[k]['Obukhov'] = (-np.abs(QC_out.loc[k]['FrictionU']**3) * np.mean(m1['T'])) / (0.4*9.81*Cwt)

# 19) Calculate stability parameter
    
            QC_out.loc[k]['ZoverL'] =  Z / QC_out.loc[k]['Obukhov']        

# 20) Do state of turbulence development (integral scale test)
            
            # theoretical value of sigma_w/ustar - parametrisation after Foken/CarboEurope
            if np.abs(QC_out.loc[k]['ZoverL']) > 1:
                sigma_uw_theory = 2
            elif np.abs(QC_out.loc[k]['ZoverL']) < 0.032:
                sigma_uw_theory = 1.3
            else:
                sigma_uw_theory = 2 * np.abs(QC_out.loc[k]['ZoverL'])**0.125  
                # For range 0.032 <= |z/l| <= 1

            w_prime = detrend(m1['w'])
            QC_out.loc[k]['sigma_w'] = np.std(w_prime)
            QC_out.loc[k]['itc_w'] = 100 * (sigma_uw_theory - (QC_out.loc[k]['sigma_w']/QC_out.loc[k]['FrictionU'])/sigma_uw_theory)

# 21) Combine stationarity test and integral scale test to get quality of flux development

            # Along wind momentum flux
            QC_out.loc[k]['quality_wu'] = flux_devel_test(QC_out.loc[k]['itc_w'], QC_out.loc[k]['sst_wu'] )
  
            # cross wind momentum flux
            QC_out.loc[k]['quality_wv'] = flux_devel_test(QC_out.loc[k]['itc_w'], QC_out.loc[k]['sst_wv'] )
        
            # Check skew/ kurtosis of w
            kurt_w = kurt_flag(QC_out.loc[k]['kurt_w'])
            skew_w = skew_flag(QC_out.loc[k]['skew_w'])

# 22) Calculate Momentum flux

            rho = np.mean(m1['rho'])
            flux_out.loc[k]['MOMFU'] = rho * eddyflux(m1['w'],m1['u'])[0]

            # QC_MOMFU - Combine all QC flags for MOMFU
            kurt_u = kurt_flag(QC_out.loc[k]['kurt_u'])
            skew_u = skew_flag(QC_out.loc[k]['skew_u'])
            if 0 in [kurt_w,skew_w,kurt_u,skew_u,QC_out.loc[k]['quality_wu']]:
                flux_out.loc[k]['QC_MOMFU'] = 0
            else:
                flux_out.loc[k]['QC_MOMFU'] = np.max([kurt_w,skew_w,kurt_u,skew_u,QC_out.loc[k]['quality_wu']])
            
            # Check momentum flux is negative (physical)
            if flux_out.loc[k]['MOMFU'] > 0.0:
                flux_out.loc[k]['QC_MOMFU'] = 0
                
# 22) Calculate Sensible heat flux and QC. 

            # SH = rho * cp * mean(w'*T')
            # rho = density of air       
            # cp = heat capacity of air
            # shf represents the loss of energy from the
            # surface by heat transfer to the atmosphere.
            # It is positive when directed away from the surface.        

            if run_SHF:
                #cp = 1003         # Specific heat capacity of dry air at constant pressure (j/kg/K) - at 250K
                cp = np.mean(m1['Cp'])
                QC_out.loc[k]['quality_wt'] = flux_devel_test(QC_out.loc[k]['itc_w'], QC_out.loc[k]['sst_wt'] )
                flux_out.loc[k]['SHF'] = rho * cp * eddyflux(m1['w'],m1['T'])[0] # W/m2
            
                # QC_SHF - Combine all QC flags for SHF
                kurt_T = kurt_flag(QC_out.loc[k]['kurt_T'])
                skew_T = skew_flag(QC_out.loc[k]['skew_T'])
                
                if flux_out.loc[k]['QC_MOMFU'] == 0:
                    flux_out.loc[k]['QC_SHF'] = 0                    
                elif 0 in [kurt_w,skew_w,kurt_T,skew_T,QC_out.loc[k]['quality_wt']]:
                    flux_out.loc[k]['QC_SHF'] = 0
                else:
                    flux_out.loc[k]['QC_SHF'] = np.max([kurt_w,skew_w,kurt_T,skew_T,QC_out.loc[k]['quality_wt']])

            else:
                print('Not enough good data for SHF %s'%str(k))
                flux_out.loc[k]['QC_SHF'] = 0    
                
# 23) Calculate Latent heat flux and QC.
            
            # Calculate latent heat flux
            # lhf = rho * L * mean(w'*q')
            # rho = density of air       
            # L = Latent heat of vaporisation
            # lhf represents the loss of energy by the
            # surface due to evaporation/ sublimation.
            # It is positive when directed away from the surface.
        
            if run_LHF:
                #L = 2264705.0     # Latent heat of vaporistaion of water (j/kg)
                L = np.mean(m1['Lv'])                
                QC_out.loc[k]['sst_wq'], Cwq, rol_cov_wq = stationarity(m1['w'],m1['q'])
                QC_out.loc[k]['quality_wq'] = flux_devel_test(QC_out.loc[k]['itc_w'], QC_out.loc[k]['sst_wq'] )        
                flux_out.loc[k]['LHF'] = L * rho * eddyflux(m1['w'],m1['q'])[0] # W/m2

                # Calculate Bowen ratio

                QC_out.loc[k]['BowenRatio'] = flux_out.loc[k]['SHF'] / flux_out.loc[k]['LHF']
    
                # QC_LHF - Combine all QC flags for LHF
                kurt_q = kurt_flag(QC_out.loc[k]['kurt_q'])
                skew_q = skew_flag(QC_out.loc[k]['skew_q'])

                if flux_out.loc[k]['QC_MOMFU'] == 0:
                    flux_out.loc[k]['QC_LHF'] = 0                  
                elif 0 in [kurt_w,skew_w,kurt_q,skew_q,QC_out.loc[k]['quality_wq']]:
                    flux_out.loc[k]['QC_LHF'] = 0
                else:
                    flux_out.loc[k]['QC_LHF'] = np.max([kurt_w,skew_w,kurt_q,skew_q,QC_out.loc[k]['quality_wq']])
            else:
                print('Not enough good data for LHF %s'%str(k))
                flux_out.loc[k]['QC_LHF'] = 0 
            
            
# 24) Get cospectral desity <wu> for ogive analysis. 
                       
            # Calculated using mathematical methods, fast fourier transform
            # FFT gives the amplitudes of variation for different frequencies
            # Calculate the FFT for x and for y. (normalised by the length of the time series)

            xw=detrend(m1['w'])
            yw=detrend(m1['u'])
            Rxx = np.fft.fft(xw)/m
            Ryy = np.fft.fft(yw)/m
        
            # The cospectrum is:
            Sxy = Rxx*np.conj(Ryy)

            # The fourier spectra are complex and symmetrical about a frequency of 0
            # (ie they have fourier components for both +ve and -ve frequency) -
            # negative frequencies don't make sense in the real world, so we 'fold'
            # the spectra about 0.

            Gxy = 2 * Sxy[1:int(m/2)]
            Gxy =np.append(Gxy, Sxy[int(m/2)+1])

            # Co-spectra: how much different sized eddies contribute to the covariance of 
            # two variables - the product of the fft of two variables. 
            # Gxx, Gyy are the power spectra of x, y, Gxy is the cospectrum of x and
            # y . The (co)spectral densities are these divided by the frequency
            # interval to give power per unit frequency. The integral of this is the
            # variance (or covariance) of the time series.

            Csdxy = Gxy / df
            cospec_out.loc[k] = Csdxy
            
            n = len(Csdxy)
            ogive = np.zeros(n)
            ogive[0]=(Csdxy*df)[0]
            for i in range(1,n):
                j=n-1-i
                ogive[j]=ogive[j+1]+(Csdxy*df)[j]

            ogive_out.loc[k] = ogive
            
               
        else:
            print('Not enough good sonic data %s'%str(k))
            flux_out.loc[k]['QC_MOMFU'] = 0
            flux_out.loc[k]['QC_LHF'] = 0 
            flux_out.loc[k]['QC_SHF'] = 0 
            
            
# 25) Save daily flux, QC and cospec<wu> estimates
            
    

    flux_out.to_csv(d_loc+'TurbulentFluxes/Flux_estimates/Flux_estimates_%smin_%s.csv'%(avp,day_str))
    QC_out.to_csv(d_loc+'TurbulentFluxes/Flux_QC/Flux_QC_%smin_%s.csv'%(avp,day_str))
    cospec_out.to_csv(d_loc+'TurbulentFluxes/Cospectra/Cospectra_wu_%smin_%s.csv'%(avp,day_str))
    ogive_out.to_csv(d_loc+'TurbulentFluxes/Cospectra/Ogive_wu_%smin_%s.csv'%(avp,day_str))