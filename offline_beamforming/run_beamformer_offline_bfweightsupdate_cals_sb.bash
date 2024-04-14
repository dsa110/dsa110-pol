#!/bin/bash
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")
corrs=("corr03" "corr04" "corr05" "corr06" "corr07" "corr08" "corr10" "corr11" "corr12" "corr14" "corr15" "corr16" "corr18" "corr19" "corr21" "corr22")
freqs=("1498.75" "1487.03125" "1475.3125" "1463.59375" "1451.875" "1440.15625" "1428.4375" "1416.71875" "1405.0" "1393.28125" "1381.5625" "1369.84375" "1358.125" "1346.40625" "1334.6875" "1322.96875")
calibdir="pol_self_calibs_FORPAPER/" #"/dataz/dsa110/operations/beamformer_weights/generated" #"/dataz/dsa110/operations/beamformer_weights/applied" #"/home/ubuntu/sherman/pol_calibs"

trigname="$3$1" 
bm="142"
mjd="$2"
echo ${trigname}
echo ${mjd}
calibdate="$5" #"$4_$5"
outdir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/$3_$4"
mkdir ${outdir}
echo $1 $2 $3 $4 $5
for i in ${!corrs[@]}; do

    dir="/media/ubuntu/ssd/sherman/$3_$4/${corrs[$i]}"
    fl="${dir}/${trigname}_data.out.sav"
    cal="${calibdir}/beamformer_weights_${sbs[$i]}_${calibdate}.dat"
    echo $fl $cal

    /home/ubuntu/proj/dsa110-shell/dsa110-xengine/src/dsaX_beamformer_offline -i ${fl} -f ${cal} -z ${freqs[$i]} -o ${bm} -a /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -p
    mv /home/ubuntu/data/tmp/output.dat ${outdir}/${corrs[$i]}_${trigname}.out

done

    
