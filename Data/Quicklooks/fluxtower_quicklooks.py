#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:03:39 2019

@author: heather
"""

from fluxtower_parse import *

# Set up plots
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams.update({'font.size': 12}) 

# Location data and plotting scripts
dpath = '/home/data/'
log_hmp = '/home/fluxtower/Quicklooks/HMP_parse_log'
log_KT = '/home/fluxtower/Quicklooks/KT_parse_log'
log_ventus =  '/home/fluxtower/Quicklooks/ventus_parse_log'
log_metek =  '/home/fluxtower/Quicklooks/metek_parse_log'
log_snd =  '/home/fluxtower/Quicklooks/snd_parse_log'
log_licor =  '/home/fluxtower/Quicklooks/licor_parse_log'

# For KT, SND: Plot one week previous from now
week_stop = dt.datetime.today()
week_start = week_stop - dt.timedelta(days=7)

# For Licor and winds, plot one day previous
day_stop = dt.datetime.today()
day_start = day_stop - dt.timedelta(days=1)

# Plot set up for weekly plots
myFmt = md.DateFormatter('%b %d')
rule = md.DayLocator(interval=1)
minor = md.HourLocator(interval=6)
fig_size = (10,6)

# KT data and plot
print('Extracting Data from KT15...')
KT = extract_KT_data(week_start,week_stop,dpath,log_KT)
dropna_KT = KT.dropna(subset=['T'])
if len(dropna_KT)!=0:
    print('Got KT15 data')
else: 
    print('!! No KT15 data !!')
    
# SnD data and plot
print('Extracting Data from SnD...')
snd = extract_snd_data(week_start,week_stop,dpath,log_snd)
dropna_snd = snd.dropna(subset=['depth'])
if len(dropna_snd)!=0:
    print('Got SnD data')
else:
    print('!! No SnD data !!')


# Snow surface (T and depth) plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(dropna_KT.index,dropna_KT['T'])
ax1.grid('on')
ax1.set_ylabel(u'Snow Temperature \N{DEGREE SIGN}C')
ax1.set_xlim(week_start,week_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(dropna_snd.index,dropna_snd['depth'])
ax2.grid('on')
ax2.set_ylabel(u'Distance from sensor (m)')
# Snd status check.
#ax1.text('SnD Status')
fig.tight_layout()
print('Saving snow surface plot.')
fig.savefig('/home/fluxtower/Quicklooks/snow_current.png')
fig.clf()


# HMP data
#print('Extracting HMP data')
#HMP1,HMP2,HMP3,HMP4,HMP5 = extract_HMP_data(week_start,week_stop,dpath,log_hmp)

# Temperature/ humidity plot
#fig = plt.figure(figsize=fig_size)
#ax1 = fig.add_subplot(211)
#ax1.plot(HMP1.index,HMP1.Ta,c='b',label='HMP1')
#ax1.plot(HMP2.index,HMP2.Ta,c='r',label='HMP2')
#ax1.plot(HMP3.index,HMP3.Ta,c='g',label='HMP3')
#ax1.plot(HMP4.index,HMP4.Ta,c='m',label='HMP4')
#ax1.grid('on')
#ax1.set_ylabel(u'T \N{DEGREE SIGN}C')
#ax1.legend(fontsize='xx-small')
#ax1.set_xlim(week_start,week_stop)
# Format ticks and labels and layout
#ax1.xaxis.set_major_locator(rule)
#ax1.xaxis.set_minor_locator(minor)
#ax1.xaxis.set_major_formatter(myFmt)

#ax2 = fig.add_subplot(212,sharex=ax1)
#ax2.plot(HMP1.index,HMP1.RH,c='b',label='HMP1')
#ax2.plot(HMP2.index,HMP2.RH,c='r',label='HMP2')
#ax2.plot(HMP3.index,HMP3.RH,c='g',label='HMP3')
#ax2.plot(HMP4.index,HMP4.RH,c='m',label='HMP4')
#ax2.set_ylabel(u'RH %')
#ax2.grid('on')
#ax2.legend(fontsize='xx-small')
#fig.tight_layout()
#print('Saving HMP plot')
#fig.savefig('/home/fluxtower/Quicklooks/T_RH_current.png')
#fig.clf()




# Plot set up for daily plots
myFmt = md.DateFormatter('%d-%H')
rule = md.HourLocator(interval=5)
minor = md.HourLocator(interval=1)

# metek data
print('Extracting Metek Data...')
m1,m2 = extract_metek_data(day_start,day_stop,dpath,log_metek)
# remove poor quality data.
m1.x = m1.x.astype(str)
m1 = m1[~m1.x.str.contains("%")]
m1.x = m1.x.astype(float)
if len(m1)!=0:
    print('Got m1 data')
else:
    print('!! No m1 data !!')

m2.x = m2.x.astype(str)
m2 = m2[~m2.x.str.contains("%")]
m2.x = m2.x.astype(float)
if len(m2)!=0:
    print('Got m2 data')
else:
    print('!! No m2 data !!')

# Metek plot
fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(m1.index,m1.x,c='r',label='x')
ax1.plot(m1.index,m1.y,c='b',label='y')
ax1.plot(m1.index,m1.z,c='g',label='z')
ax1.grid('on')
ax1.set_ylabel('M1 WSD')
ax1.legend(fontsize='xx-small')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.plot(m2.index,m2.x,c='r',label='x')
ax2.plot(m2.index,m2.y,c='b',label='y')
ax2.plot(m2.index,m2.z,c='g',label='z')
ax2.grid('on')
ax2.set_ylabel('M2 WSD')
ax2.legend(fontsize='x-small')

# Format ticks and labels and layout
ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Metek plot...')
fig.savefig('/home/fluxtower/Quicklooks/metek_current.png')
fig.clf()




# ventus data
print('Extracting Ventus data..')
v1,v2 = extract_ventus_data(day_start,day_stop,dpath,log_ventus)

# filter nans
v1 = v1.dropna()
v2 = v2.dropna()

if len(v1)!=0:
    print('Got V1 data')
else:
    print('!! No V1 data !!')
if len(v2)!=0:
    print('Got V2 data')
else:
    print('!! No V2 data !!')

v1.wdir = v1.wdir.astype(str)
v1 = v1[~v1.wdir.str.contains("F")]
v1.wdir = v1.wdir.astype(int)
v2.wdir = v2.wdir.astype(str)
v2 = v2[~v2.wdir.str.contains("F")]
v2.wdir = v2.wdir.astype(int)

v1.wsd = v1.wsd.astype(str)
v1.wsd = v1.wsd.str.lstrip('\x03\x02')
v1.wsd = v1.wsd.astype(float)
v2.wsd = v2.wsd.astype(str)
v2.wsd = v2.wsd.str.lstrip('\x03\x02')
v2.wsd = v2.wsd.astype(float)

 # ventus plot

fig = plt.figure(figsize=fig_size)
ax1 = fig.add_subplot(211)
ax1.plot(v1.index,v1.wsd,c='b',label='v1')
ax1.plot(v2.index,v2.wsd,c='r',label='v2')
ax1.grid('on')
ax1.set_ylabel('WSD')
ax1.legend(fontsize='x-small')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.scatter(v1.index,v1.wdir,c='b',label='v1')
ax2.scatter(v2.index,v2.wdir,c='r',label='HMP2')
ax2.grid('on')
ax2.set_ylim(0,360)
ax2.set_ylabel('WDIR')

# Format ticks and labels and layout
ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Ventus plot..')
fig.savefig('/home/fluxtower/Quicklooks/ventus_current.png')
fig.clf()




# Licor data
#start=datetime.datetime(2019, 5, 19, 16, 49, 34, 873137)
#stop = datetime.datetime(2019, 5, 20, 16, 49, 34, 873137)
print('Extracting Licor data...')
licor = extract_licor_data(day_start,day_stop,dpath,log_licor)
if len(licor)!=0:
    print('Got Licor data')
else:
    print('!! No Licor data !!')

# Licor plot

fig = plt.figure(figsize=fig_size)

#H2O density (mmol m-3)
ax1 = fig.add_subplot(211)
ax1.plot(licor.index,licor.H2OD,c='b')
ax1.grid('on')
ax1.set_ylabel('H2O density (mmol m-3)')

ax2 = fig.add_subplot(212,sharex=ax1)
ax2.scatter(licor.index,licor.CO2D,c='r')
ax2.grid('on')
ax2.set_ylabel('CO2 density (mmol m-3)')

# Format ticks and labels and layout
#ax1.set_xlim(day_start,day_stop)
ax1.xaxis.set_major_locator(rule)
ax1.xaxis.set_minor_locator(minor)
ax1.xaxis.set_major_formatter(myFmt)
plt.setp(ax1.get_xticklabels(), visible=False)
fig.tight_layout()
print('Saving Licor plot...')
fig.savefig('/home/fluxtower/Quicklooks/licor_current.png')
fig.clf()

print('Quicklook plots updated, end of script.')
