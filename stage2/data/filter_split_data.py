import glob
import os
import pickle
import shutil
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm




def split_data(keypoint_folder, wav_folder):
    # train - test split
    train_list = sorted([Path(f).name[:-5] for f in Path(f"{keypoint_folder}/train").glob(f"*.*") if Path(f).is_file()])
    test_list = sorted([Path(f).name[:-5] for f in Path(f"{keypoint_folder}/test").glob(f"*.*") if Path(f).is_file()])

    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{split_name}/keypoints").mkdir(parents=True, exist_ok=True)
        Path(f"{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        
        count = 0
        for sequence in tqdm(split_list):
            keypoint = f"{keypoint_folder}/{split_name}/{sequence}.json"
            wav = f"{wav_folder}/{sequence}.wav"
            if Path(keypoint).exists() and Path(wav).exists():
                with open(keypoint) as f:
                    sample_dict = json.loads(f.read())
                    np_keypoints = np.array(sample_dict['kp']) # L,100,2
                    seq_len = np_keypoints.shape[0]
                    np_keypoints = np_keypoints.reshape(seq_len, -1) # L,200
                    np.save(f"{split_name}/keypoints/{sequence}.npy", np_keypoints)
                    shutil.copyfile(wav, f"{split_name}/wavs/{sequence}.wav")
                    count += 1
            else:
                print(f"{sequence} not found")
                
        print(f"{split_name} count: {count}")
                    
