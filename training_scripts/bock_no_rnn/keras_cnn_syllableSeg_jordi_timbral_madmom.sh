#!/bin/bash

# change python version
#module load cuda/8.0
#module load theano/0.8.2
#module load python/2.7.5
#module load essentia/2.1_python-2.7.5

# two variables you need to set
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.475
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate keras_env


#$ -N tim_jordi
#$ -q default.q
#$ -l h=node07

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/schluter_jordi_tim_madmom.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/schluter_jordi_tim_madmom.$JOB_ID.err

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter
mkdir /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter/syllableSeg

#printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
#start=`date +%s`
#cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_artist_filter_split.h5 /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter/syllableSeg/
#end=`date +%s`

#printf "Finish copying feature files into scratch directory...\n"
#printf $((end-start))

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsSchluter/keras_cnn_syllableSeg_jordi_madmom.py 1 0 8 jordi_timbral_schluter

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_part1_jordi_timbral_schluter
fi
printf "Job done. Ending at `date`\n"
