#!/bin/bash

corrs=("corr03" "corr04" "corr05" "corr06" "corr07" "corr08" "corr10" "corr11" "corr12" "corr14" "corr15" "corr16" "corr18" "corr19" "corr21" "corr22")
sbs=("sb00" "sb01" "sb02" "sb03" "sb04" "sb05" "sb06" "sb07" "sb08" "sb09" "sb10" "sb11" "sb12" "sb13" "sb14" "sb15")
freqs=("1498.75" "1487.03125" "1475.3125" "1463.59375" "1451.875" "1440.15625" "1428.4375" "1416.71875" "1405.0" "1393.28125" "1381.5625" "1369.84375" "1358.125" "1346.40625" "1334.6875" "1322.96875")

#Command line arguments
#1 == Datestring
#2 == Candname (e.g. ane)
#3 == nickname (e.g. 3C48)
#4 == Calibrator name _ date
#5 == Beam
#6 == MJD
#7 == DM

echo $1 $2 $3 $4 $5 $6 $7

calibdir="/media/ubuntu/ssd/sherman/code/pol_self_calibs_FORPAPER/" #"/dataz/dsa110/operations/beamformer_weights/generated" #"/dataz/dsa110/operations/beamformer_weights/applied" #"/home/ubuntu/sherman/pol_calibs"
#dir="/dataz/dsa110/candidates/$2/Level2/voltages/" #"/media/ubuntu/data/dsa110/T3/$1"
outdir="/media/ubuntu/ssd/sherman/scratch_weights_update_2022-06-03_32-7us/$3_$8"
deldir="3C286_delay.yaml" #"/dataz/dsa110/operations/beamformer_weights/applied/beamformer_weights_$4.yaml"
mkdir ${outdir}
echo $calibdir $dir $outdir
calibdate="$4" #"$4T00:00:00"
trigname="$3$2" 
#calibname="J043236+413829"
bm="$5"
mjd="$6"
DM="$7"
tbin="4096" #"4"
nsamps="40960" #"5120"
start_offset="0"
stokes="0"
minbase="0.0"

for i in ${!corrs[@]}; do

    dir="/media/ubuntu/ssd/sherman/$3_$8/${corrs[$i]}"
    fl="${dir}/${trigname}_data.out.sav" #${corrs[$i]}_$3$2.out" #${trigname}_${sbs[$i]}_data.out"
    cal="${calibdir}/beamformer_weights_${sbs[$i]}_${calibdate}.dat"
    echo $fl $cal

    /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit_dev -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} -n
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f flagants_test.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}

    echo "AFTER TOOLKIT"
done

    
python /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/splicer_dev.py ${mjd} ${bm} ${outdir}/${trigname}_dev_${stokes}.fil ${tbin} ${outdir}/corr03.out ${outdir}/corr04.out ${outdir}/corr05.out ${outdir}/corr06.out ${outdir}/corr07.out ${outdir}/corr08.out ${outdir}/corr10.out ${outdir}/corr11.out ${outdir}/corr12.out ${outdir}/corr14.out ${outdir}/corr15.out ${outdir}/corr16.out ${outdir}/corr18.out ${outdir}/corr19.out ${outdir}/corr21.out ${outdir}/corr22.out


stokes="1"
minbase="0.0"

for i in ${!corrs[@]}; do

    dir="/media/ubuntu/ssd/sherman/$3_$8/${corrs[$i]}"
    fl="${dir}/${trigname}_data.out.sav" #${corrs[$i]}_$3$2.out" #${trigname}_${sbs[$i]}_data.out"
    cal="${calibdir}/beamformer_weights_${sbs[$i]}_${calibdate}.dat"
    echo $fl $cal

    /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit_dev -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} -n
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f flagants_test.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}

    echo "AFTER TOOLKIT"
done


python /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/splicer_dev.py ${mjd} ${bm} ${outdir}/${trigname}_dev_${stokes}.fil ${tbin} ${outdir}/corr03.out ${outdir}/corr04.out ${outdir}/corr05.out ${outdir}/corr06.out ${outdir}/corr07.out ${outdir}/corr08.out ${outdir}/corr10.out ${outdir}/corr11.out ${outdir}/corr12.out ${outdir}/corr14.out ${outdir}/corr15.out ${outdir}/corr16.out ${outdir}/corr18.out ${outdir}/corr19.out ${outdir}/corr21.out ${outdir}/corr22.out

stokes="2"
minbase="0.0"

for i in ${!corrs[@]}; do

    dir="/media/ubuntu/ssd/sherman/$3_$8/${corrs[$i]}"
    fl="${dir}/${trigname}_data.out.sav" #${corrs[$i]}_$3$2.out" #${trigname}_${sbs[$i]}_data.out"
    cal="${calibdir}/beamformer_weights_${sbs[$i]}_${calibdate}.dat"
    echo $fl $cal

    /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit_dev -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} -n
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f flagants_test.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}


    echo "AFTER TOOLKIT"
done


python /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/splicer_dev.py ${mjd} ${bm} ${outdir}/${trigname}_dev_${stokes}.fil ${tbin} ${outdir}/corr03.out ${outdir}/corr04.out ${outdir}/corr05.out ${outdir}/corr06.out ${outdir}/corr07.out ${outdir}/corr08.out ${outdir}/corr10.out ${outdir}/corr11.out ${outdir}/corr12.out ${outdir}/corr14.out ${outdir}/corr15.out ${outdir}/corr16.out ${outdir}/corr18.out ${outdir}/corr19.out ${outdir}/corr21.out ${outdir}/corr22.out


stokes="3"
minbase="0.0"

for i in ${!corrs[@]}; do

    dir="/media/ubuntu/ssd/sherman/$3_$8/${corrs[$i]}"
    fl="${dir}/${trigname}_data.out.sav" #${corrs[$i]}_$3$2.out" #${trigname}_${sbs[$i]}_data.out"
    cal="${calibdir}/beamformer_weights_${sbs[$i]}_${calibdate}.dat"
    echo $fl $cal

    /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit_dev -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} -n
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f /home/ubuntu/proj/dsa110-shell/dsa110-xengine/scripts/flagants.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}
    #/home/ubuntu/proj/dsa110-shell/dsa110-bbproc/toolkit -i ${fl} -w ${cal} -p ${outdir}/${corrs[$i]}.out -c ${freqs[$i]} -b ${bm} -f flagants_test.dat -t ${tbin} -m ${DM} -s ${nsamps} -q ${start_offset} -g ${stokes} -v ${minbase} #-d ${deldir}


    echo "AFTER TOOLKIT"
done


python /home/ubuntu/proj/dsa110-shell/dsa110-bbproc/splicer_dev.py ${mjd} ${bm} ${outdir}/${trigname}_dev_${stokes}.fil ${tbin} ${outdir}/corr03.out ${outdir}/corr04.out ${outdir}/corr05.out ${outdir}/corr06.out ${outdir}/corr07.out ${outdir}/corr08.out ${outdir}/corr10.out ${outdir}/corr11.out ${outdir}/corr12.out ${outdir}/corr14.out ${outdir}/corr15.out ${outdir}/corr16.out ${outdir}/corr18.out ${outdir}/corr19.out ${outdir}/corr21.out ${outdir}/corr22.out

