#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_pretrained_schluter ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_pretrained_schluter
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_pretrained_schluter
mkdir /scratch/rgongcnnSyllableSeg_pretrained_schluter/syllableSeg


printf "Copying feature files into scratch directory...\n"
# Third, copy the experiment's data:
# ----------------------------------
start=`date +%s`
cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_artist_filter_madmom.h5 /scratch/rgongcnnSyllableSeg_pretrained_schluter/syllableSeg/
end=`date +%s`

printf "Finish copying feature files into scratch directory...\n"
printf $((end-start))


#$ -N art_sch
#$ -q default.q
#$ -l h=node10

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/cnn_jingju_deep_pretrained.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/cnn_jingju_deep_pretrained.$JOB_ID.err

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsTransferLearning/keras_cnn_syllableSeg_less_deep_pretrained_schluter.py

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_pretrained_schluter ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_pretrained_schluter
fi
printf "Job done. Ending at `date`\n"
