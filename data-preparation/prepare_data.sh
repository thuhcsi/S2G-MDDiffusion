#!/bin/bash

speakers=("chem" "oliver" "jon" "seth")

mkdir raw_data
mkdir crop_data
mkdir data

for speaker in "${speakers[@]}"
do  
    # download raw videos
    # |--- raw_data
    # |    |--- chem
    # |    |    |--- XXXXX.mp4
    # |    |--- oliver
    # |    |--- jon
    # |    |--- seth
    cd raw_data
    mkdir $speaker && cd $speaker
    python ../../download_videos.py --speaker $speaker --meta_path ../../cmu_intervals_df.csv --filter_path ../../filtered_intervals.json
    cd ../..

    # trim videos (temporally)
    # |--- trim_data
    # |    |--- chem
    # |    |    |--- chemistry#99999.mp4
    # |    |--- oliver
    # |    |--- jon
    # |    |--- seth
    python trim_videos.py --speaker $speaker

    # crop videos (spatially) && convert to frame images && split && extract audio from trimed mp4 data
    # |--- data
    # |    |--- img
    # |    |    |--- train
    # |    |    |    |--- chemistry#99999.mp4
    # |    |    |    |--- oliver#88888.mp4
    # |    |    |--- test
    # |    |    |    |--- jon#77777.mp4
    # |    |    |    |--- seth#66666.mp4
    # |    |--- audio
    # |    |    |--- chemistry#99999.mp4
    # |    |    |--- oliver#88888.mp4
    # |    |    |--- jon#77777.mp4
    # |    |    |--- seth#66666.mp4
    python crop_videos.py --speaker $speaker

done
