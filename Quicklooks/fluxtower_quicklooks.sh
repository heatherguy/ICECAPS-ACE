#!/bin/sh

/home/fluxtower/anaconda3/bin/python /home/fluxtower/Quicklooks/metek_quicklook.py > /home/fluxtower/Quicklooks/metek_plot_log.txt

/home/fluxtower/anaconda3/bin/python /home/fluxtower/Quicklooks/ventus_quicklook.py > /home/fluxtower/Quicklooks/ventus_plot_log.txt

/home/fluxtower/anaconda3/bin/python /home/fluxtower/Quicklooks/HMP_quicklook.py >/home/fluxtower/Quicklooks/HMP_plot_log.txt

/home/fluxtower/anaconda3/bin/python /home/fluxtower/Quicklooks/snow_quicklook.py >/home/fluxtower/Quicklooks/snow_plot_log.txt

/home/fluxtower/anaconda3/bin/python /home/fluxtower/Quicklooks/licor_quicklook.py >/home/fluxtower/Quicklooks/licor_plot_log.txt
