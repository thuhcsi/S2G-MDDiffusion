import argparse

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.wavlm_features import extract_folder as wavlm_extract
from filter_split_data import *
from slice import *

def create_dataset(opt):
    # copy and split the features and wavs
    print("Creating train / test split")
    split_data(opt.keypoint_folder, opt.wav_folder)
    
    # extract full wavlm featues，saved in data/train/wavlm_feats and data/test/wavlm_feats
    if opt.extract_wavlm:
        print("Extracting wavlm features")
        wavlm_extract("train/wavs", "train/wavlm_feats")
        wavlm_extract("test/wavs", "test/wavlm_feats")
    
    # slice motion features, wavs, wavlm features
    print("Slicing train data")
    slice_data(f"train/keypoints", f"train/wavs", f"train/wavlm_feats",stride=opt.stride, length=opt.length)
    print("Slicing test data")
    slice_data(f"test/keypoints", f"test/wavs", f"test/wavlm_feats", stride=opt.stride, length=opt.length)
    
    # extract baseline features for sliced wavs
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract("train/wavs_sliced", "train/baseline_feats_sliced")
        baseline_extract("test/wavs_sliced", "test/baseline_feats_sliced")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.4)
    parser.add_argument("--length", type=float, default=3.2, help="checkpoint")
    parser.add_argument(
        "--keypoint_folder",
        type=str,
        default=None,
        help="folder containing tps features",
    )
    parser.add_argument(
        "--wav_folder",
        type=str,
        default=None,
        help="folder containing audio",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-wavlm", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
