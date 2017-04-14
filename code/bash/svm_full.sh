#!/bin/sh
#PBS -l walltime=14:00:00
#PBS -l select=1:ncpus=1:mem=128gb
#PBS -N RFS

echo "Beginning script..."

module load anaconda/2.4.1

echo "Modules loaded"

cp $HOME/project/RFS.py $TMPDIR
cp $WORK/project/DataStore.h5 $TMPDIR

echo "Files copied"

source activate FYP

echo "Anaconda environment activated"

python RFS.py

echo "Python file ran"

cp linear_all_results.txt $WORK/project

echo "Output copied"

source deactivate

module purge
