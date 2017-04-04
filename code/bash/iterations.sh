#!/bin/sh

declare -a ids

START=19
END=24

for ((i=${START}; i<${END}; i++))
do
    mkdir $WORK/project/iteration_${i}_results
    index=${i}-${START}	
	
    if [ $i -gt ${START} ]
    then
        prevIndex=${index}-1
        var=${ids[${prevIndex}]}
        ID=$(qsub -v IT=${i} -W depend=afterany:${var} $TMPDIR/feature_serial.sh)
        qsub -v IT=${i} -W depend=afterany:${ID} read_files.sh
    fi
	
    if [ $i -eq ${START} ]
    then
        ID=$(qsub -v IT=${i} $TMPDIR/feature_serial.sh)
        qsub -v IT=${i} -W depend=afterany:${ID} read_files.sh
    fi
    
    ids[${index}]=$ID
done

echo ${ids[*]}

