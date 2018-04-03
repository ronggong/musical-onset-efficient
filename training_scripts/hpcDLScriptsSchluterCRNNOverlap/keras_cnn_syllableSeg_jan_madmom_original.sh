#!/bin/bash

# two variables you need to set
#device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH

#export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.475
#export PATH=/usr/local/cuda/bin/:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
#
#export LD_LIBRARY_PATH=/soft/cuda/cudnn/8.0/lib64:$LD_LIBRARY_PATH
#export CPATH=/soft/cuda/cudnn/8.0/include:$CPATH
#export LIBRARY_PATH=/soft/cuda/cudnn/8.0/lib64:$LD_LIBRARY_PATH

source activate keras_env

#$ -N sch_bd_200
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/schluter_jan_phrase_overlap.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/schluter_jan_phrase_overlap.$JOB_ID.err

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap
mkdir /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap/syllableSeg

#printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
start=`date +%s`
cp -rp /homedtic/rgong/cnnSyllableSeg/schluter/feature_madmom_simpleSampleWeighting_phrase/* /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap/syllableSeg/
end=`date +%s`

printf "Finish copying feature files into scratch directory...\n"
printf $((end-start))
printf "\n"

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsSchluterCRNNOverlap/keras_cnn_syllableSeg_jan_madmom_original.py 0 8 0

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_phrase_schluter_overlap
fi
printf "Job done. Ending at `date`\n"
