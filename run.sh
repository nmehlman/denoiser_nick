#!/bin/bash
stage=1


source /data/home/anirudh/anaconda3/etc/profile.d/conda.sh

conda activate denoise

if [ $stage -eq 1 ]; then

python train.py

fi
