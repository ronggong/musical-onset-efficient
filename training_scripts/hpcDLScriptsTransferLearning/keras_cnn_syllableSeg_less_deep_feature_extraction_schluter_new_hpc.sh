#!/bin/bash

#SBATCH -J jj_fea
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation
#SBATCH --gres=gpu:maxwell:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=1
#SBATCH --threads-per-core=1

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/cnnSyllableSeg/out/jingju_fea.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/cnnSyllableSeg/out/jingju_fea.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/hpcDLScriptsTransferLearning/keras_cnn_syllableSeg_less_deep_feature_extraction_schluter_new_hpc.py