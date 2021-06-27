#!/bin/bash

#get config for data prep
. conf/data.config

# source environment
source /data/home/anirudh/anaconda3/etc/profile.d/conda.sh

conda activate denoise

# Perform noise addition to the clean files at varied SNRs
if [ $stage -eq 1 ]; then
	# noise/ is the path to the white gaussian and pink noise audio files
        echo "Noisy data = $noisy_path, Clean data = $clean_path"
        python denoiser/augment_noise.py noise/ $clean_path

fi


if [ $stage -eq 2 ]; then

echo "egs_paths = $egs_path_tr, $egs_path_val, $egs_path_tst"
# Create Json files for the data in egs folder for train. 
	path=$egs_path_tr
	if [[ ! -e $path ]]; then
    		mkdir -p $path
	fi
	python3 -m denoiser.audio  $clean_path > $path/clean.json
	python3 -m denoiser.audio  $noisy_path > $path/noisy.json

exit
# Create Json files for the data in egs folder for validation.
        path=$egs_path_val
        if [[ ! -e $path ]]; then
                mkdir -p $path
        fi
        python3 -m denoiser.audio  $clean_path > $path/clean.json
        python3 -m denoiser.audio  $noisy_path > $path/noisy.json

# Create Json files for the data in egs folder for test.
        path=$egs_path_tst
        if [[ ! -e $path ]]; then
                mkdir -p $path
        fi
        python3 -m denoiser.audio  $clean_path > $path/clean.json
        python3 -m denoiser.audio  $noisy_path > $path/noisy.json
fi

