#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:32:22 2018

@author: eehgu
"""

# Check yesterdays data uploaded at 1500 each day

import os
from ftplib import FTP
from datetime import datetime
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from os.path import basename



def email_me(address_list,message,subject, files=None):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.ehlo()
    server.login("heatherspython@gmail.com", "Phati&Riti")
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'ACE Daily Data Check'
    msg['To'] = ''
    msg.attach(MIMEText(message))
    
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)
    
    
    for i in address_list:
        server.sendmail("heatherspython@gmail.com", i, msg.as_string())
    server.quit()
        
        


# See if yesterdays data has uploaded
yesterday = (datetime.utcnow() - timedelta(hours=24)).date()

opc_path = '/psd3/arctic/summit/opc/quicklooks/'
opc_fname = 'OPC_%s.png'%str(yesterday)

#counts_path = '/psd3/arctic/summit/cpc/quicklooks/'
#counts_fname = 'Ncounts_%s.png'%str(yesterday)



ftp = FTP("ftp1.esrl.noaa.gov", "anonymous", "eehgu@leeds.ac.uk")
ftp.cwd(opc_path)

if opc_fname in ftp.nlst():
    # Save the OPC plot
    local_filename = os.path.join(r"/Users/eehgu/Desktop/opc", opc_fname)
    lf = open(local_filename, "wb")
    ftp.retrbinary("RETR " + opc_fname, lf.write)
    lf.close()
    ftp.cwd(cpc_path)
    # All data is there - email all the files. 
    email_me(['eehgu@leeds.ac.uk'],'OPC data downloaded, quickview attached.','Todays ACE data',files =['/Users/eehgu/Desktop/opc/%s'%opc_fname])
else:
    # We're missing the opc data
    email_me(['eehgu@leeds.ac.uk'],'The OPC quicklook did not update today. Better check it out.','ACE DATA ISSUE')

    
    


# If it has not, email me to let me know. 