import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class GestureDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        backup_path: str,
        train: bool,
        feature_type: str = "wavlm",
        normalizer: Any = None,
        data_len: int = -1,
        include_contacts: bool = True,
        force_reload: bool = False,
    ):
        self.data_path = data_path

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.feature_type = feature_type

        self.normalizer = normalizer
        self.data_len = data_len

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        data = self.load_data() 

        print(
            f"Loaded {self.name} Dataset With Dimensions: Keypoints: {data['keypoints'].shape}, Wav_features: {data['wav_features'].shape}, Wavs: {len(data['wavs'])}"
        )

        self.data = {
            "keypoints": data['keypoints'],
            "wav_features": data["wav_features"],
            "wavs": data["wavs"],
        }
        assert len(data['keypoints']) == len(data["wav_features"])
        self.length = len(data['keypoints'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_feature = torch.from_numpy(self.data["wav_features"][idx])
        keypoint = self.data['keypoints'][idx]
        keypoint_cond = torch.from_numpy(keypoint[0, :].astype(np.float32))
        keypoint_input = torch.from_numpy(keypoint[1:, :].astype(np.float32))
        return keypoint_input, keypoint_cond, wav_feature, self.data["wavs"][idx]

    def load_data(self):
        split_data_path = os.path.join(
            self.data_path, "train" if self.train else "test"
        )

        # Structure:
        # data
        #   |- train
        #   |    |- keypoints_sliced
        #   |    |- wav_sliced
        #   |    |- baseline_feats_sliced
        #   |    |- wavlm_feats_sliced
        #   |    |- keypoints
        #   |    |- wavs
        #   |    |- wavlm_feats

        keypoint_path = os.path.join(split_data_path, "keypoints_sliced")
        feature_path = os.path.join(split_data_path, f"{self.feature_type}_feats_sliced")
        baseline_path = os.path.join(split_data_path, f"baseline_feats_sliced")
        wav_path = os.path.join(split_data_path, f"wavs_sliced")
        # sort keypoints and sounds
        keypoints = sorted(glob.glob(os.path.join(keypoint_path, "*.npy")))
        features = sorted(glob.glob(os.path.join(feature_path, "*.npy")))
        baseline_features = sorted(glob.glob(os.path.join(baseline_path, "*.npy")))
        wavs = sorted(glob.glob(os.path.join(wav_path, "*.wav")))

        # stack the keypoints and features together
        all_keypoints = []
        all_features = []
        all_wavs = []
        assert len(keypoints) == len(features) == len(baseline_features) == len(wavs)
        for keypoint, feature, baseline_feature, wav in tqdm(zip(keypoints, features, baseline_features, wavs)):
            # make sure name is matching
            k_name = os.path.splitext(os.path.basename(keypoint))[0]
            f_name = os.path.splitext(os.path.basename(feature))[0]
            w_name = os.path.splitext(os.path.basename(wav))[0]
            assert k_name == f_name == w_name, str((keypoint, feature, wav))
            # load keypoints
            data = np.load(keypoint)
            all_keypoints.append(data)
            if self.feature_type != 'baseline':
                input_feature = np.concatenate((np.load(feature), np.load(baseline_feature)), axis=-1)
            else:
                input_feature = np.load(baseline_feature)
            all_features.append(input_feature)
            all_wavs.append(wav)

        all_keypoints = np.array(all_keypoints)  
        all_features = np.array(all_features) 

        print(all_keypoints.shape)
        print(all_features.shape)
        
        data = {"keypoints": all_keypoints, "wav_features": all_features, "wavs": all_wavs}

        return data



