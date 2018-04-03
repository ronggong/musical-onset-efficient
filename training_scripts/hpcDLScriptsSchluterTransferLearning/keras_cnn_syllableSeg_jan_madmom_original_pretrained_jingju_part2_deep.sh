#!/bin/bash

export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate keras_env

#$ -N feat_ext_deep
#$ -q default.q
#$ -l debian8
#$ -l h=node11

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/schluter_jan_transfer.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/schluter_jan_transfer.$JOB_ID.err

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_part2_jan ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_part2_jan
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_part2_jan
mkdir /scratch/rgongcnnSyllableSeg_part2_jan/syllableSeg

##printf "Copying feature files into scratch directory...\n"
## Third, copy the experiment's data:
## ----------------------------------
#start=`date +%s`
#cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/feature_all_artist_filter_madmom.h5 /scratch/rgongcnnSyllableSeg_part2_jan/syllableSeg
#end=`date +%s`
#
#printf "Finish copying feature files into scratch directory...\n"
#printf $((end-start))

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsSchluterTransferLearning/keras_cnn_syllableSeg_jan_madmom_original_pretrained_jingju_deep.py 2 0 8 deep

# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_part2_jan ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_part2_jan
fi
printf "Job done. Ending at `date`\n"
