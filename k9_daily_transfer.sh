#!/bin/bash

yesterday=`date -d "1 day ago" '+%Y-%m-%d'`

rsync -az /home/fluxtower/Data/CPC_Summit_"$yesterday".csv k9:/home/earrn/ICECAPS_ACE_Backup/.

#rsync -az /home/fluxtower/Data/CLASP_F_Summit_"$yesterday".csv k9:/home/earrn/ICECAPS_ACE_Backup/.

rsync -az /home/fluxtower/Data/OPC k9:/home/earrn/ICECAPS_ACE_Backup/.

daybeforeyesterday=`date -d "2 days ago" '+%Y-%m-%d'`

rsync -az /home/fluxtower/Data/CPC_Summit_"$daybeforeyesterday".csv k9:/home/earrn/ICECAPS_ACE_Backup/.

#rsync -az /home/fluxtower/Data/CLASP_F_Summit_"$daybeforeyesterday".csv k9:/home/earrn/ICECAPS_ACE_Backup/.
