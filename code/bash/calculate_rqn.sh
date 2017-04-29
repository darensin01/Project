#!/bin/sh
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -N RQN_100

echo "Beginning script..."

module load anaconda/2.4.1

echo "Modules loaded"

cp $HOME/project/calculate_rqn.py $TMPDIR
cp $WORK/project/DataStore.h5 $TMPDIR

echo "Files copied"

source activate FYP

echo "Anaconda environment activated"

python calculate_rqn.py

echo "Python file ran"

cp mrmmc_results_100.txt $WORK/project

echo "Output copied"

source deactivate

module purge
