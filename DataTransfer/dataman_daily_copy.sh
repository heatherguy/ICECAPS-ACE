#!/bin/bash

yesterday=`date -d "1 day ago" '+%Y-%m-%d'`

cp /home/fluxtower/Data/OPC/OPC_"$yesterday".txt /samba/ACE_To_Boulder/.

cp /home/fluxtower/Data/OPC_plot_archive/OPC_"$yesterday".png /samba/ACE_To_Boulder/.

cp /home/fluxtower/Data/Ncounts_plot_archive/Ncounts_"$yesterday".png /samba/ACE_To_Boulder/.
