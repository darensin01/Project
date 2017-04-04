#!/bin/sh
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=1:mem=64gb
#PBS -N FS_SERIAL

#PBS -J 0-5

ITERATION=${IT}

echo "Beginning script..."

module load anaconda/2.4.1

echo "Modules loaded"

cp $WORK/project/greedy_feature_selection.py $TMPDIR
cp $WORK/project/DataStore.h5 $TMPDIR

for ((i=2; i<ITERATION; i++))
do
    cp $WORK/project/iteration_${i}_results/all_${i}_feature_results.txt $TMPDIR
done

echo "Files copied"

source activate FYP

echo "Anaconda environment activated"

python greedy_feature_selection.py $PBS_ARRAY_INDEX $ITERATION

echo "Python file ran"

cp ${ITERATION}_feature_selection_results_$PBS_ARRAY_INDEX.txt $WORK/project/iteration_${ITERATION}_results

echo "Output copied"

source deactivate

module purge
