#!/bin/bash

# change python version
#module load cuda/8.0
#module load theano/0.8.2
#module load python/2.7.5
#module load essentia/2.1_python-2.7.5

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate keras_env


#$ -N sseg_tim
#$ -q default.q
#$ -l h=node10

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/schluter_timbral.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/schluter_timbral.$JOB_ID.err

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_timbral ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_timbral
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_timbral
mkdir /scratch/rgongcnnSyllableSeg_timbral/syllableSeg

#printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
#start=`date +%s`
#cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_artist_filter_split.h5 /scratch/rgongcnnSyllableSeg_timbral/syllableSeg/
#end=`date +%s`

#printf "Finish copying feature files into scratch directory...\n"
#printf $((end-start))

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsSchluter/keras_cnn_syllableSeg_conv_dense_timbral_filters_jordi.py

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_timbral ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_timbral
fi
printf "Job done. Ending at `date`\n"
