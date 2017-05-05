#!/bin/sh
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=1:mem=200gb
#PBS -N T_TEST

echo "Beginning script..."

module load anaconda/2.4.1

echo "Modules loaded"

cp $HOME/project/t_test.py $TMPDIR
cp $WORK/project/DataStore.h5 $TMPDIR

echo "Files copied"

source activate FYP

echo "Anaconda environment activated"

python t_test.py

echo "Python file ran"

cp t_test_results_no_parallel.txt $WORK/project

echo "Output copied"

source deactivate

module purge
