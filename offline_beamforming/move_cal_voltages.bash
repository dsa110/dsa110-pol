#!/bin/bash

#copies voltages from T3 to specified directory
# $1 -- calibrator name (e.g. 3C48, 3C286)
# $2 -- observation id (e.g. xxu, omu)
# $3 -- caldate (e.g. Sun_Dec_18_2022)
# $4 -- path to new directory
# $5 -- path to T3 directory


name="$1$2"
corrs=("corr03" "corr04" "corr05" "corr06" "corr07" "corr08" "corr10" "corr11" "corr12" "corr14" "corr15" "corr16" "corr18" "corr19" "corr21" "corr22")
outdir="$4$3/$1"
initdir="$5"

mkdir "$4$3"
mkdir $outdir #3C48_2023-01-30
echo $1 $2 $3 $4 $5

for j in ${!corrs[@]}
do
	echo ${corrs[$j]}
	echo ${outdir}/${corrs[$j]}
	#mkdir $outdir/${corrs[$j]}
	
	echo ${initdir}/${corrs[$j]}/${name}
	ls ${initdir}/${corrs[$j]}/${name}*
	#mv ${initdir}/${corrs[$j]}/${name}* ${outdir}/${corrs[$j]}
	echo " "
done
