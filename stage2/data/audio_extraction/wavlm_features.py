import os
from functools import partial
from pathlib import Path

import librosa
import librosa as lr
import numpy as np
from tqdm import tqdm
import torch
import sys
sys.path.append('./wavlm')
sys.path.append('./data/wavlm')
from WavLM import WavLM, WavLMConfig
import torch.nn.functional as F
import math

FPS = 25
HOP_LENGTH = 512
SR = FPS * HOP_LENGTH
EPS = 1e-6


def _get_tempo(audio_name):
    """Get tempo (BPM) for a music by parsing music name."""

    # a lot of stuff, only take the 5th element
    audio_name = audio_name.split("_")[4]

    assert len(audio_name) == 4
    if audio_name[0:3] in [
        "mBR",
        "mPO",
        "mLO",
        "mMH",
        "mLH",
        "mWA",
        "mKR",
        "mJS",
        "mJB",
    ]:
        return int(audio_name[3]) * 10 + 80
    elif audio_name[0:3] == "mHO":
        return int(audio_name[3]) * 5 + 110
    else:
        assert False, audio_name


def extract(fpath, skip_completed=True, dest_dir="aist_baseline_feats"):
    
    # extract wavlm
    # load the pre-trained checkpoints
    device = torch.device("cuda")
    checkpoint = torch.load('wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # extract the representation of last layer
    wav_input_16khz, _ = librosa.load(fpath, sr=16000)
    wav_input_16khz = torch.from_numpy(wav_input_16khz).to(device)
    print(wav_input_16khz.shape[0]/16000*25)
    wav_input_16khz = wav_input_16khz.unsqueeze(0)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    wavlm_feats = model.extract_features(wav_input_16khz)[0]
    wavlm_feats = wavlm_feats.detach().cpu()
    
    last_feature = wavlm_feats[:, -1, :].unsqueeze(1)
    wavlm_feats = torch.cat((wavlm_feats, last_feature), dim=1)
    print(wavlm_feats.shape)
    
    wavlm_feats = F.interpolate(wavlm_feats.transpose(1, 2), size=math.ceil(wavlm_feats.shape[1] / 2), align_corners=True,
                            mode='linear').transpose(1, 2).squeeze(0)   # 50fps -> 25fps
    wavlm_feats = wavlm_feats.numpy()
    
    # extract baseline
    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return
    return wavlm_feats, save_path


def extract_folder(src, dest):
    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    model, cfg = wavlm_init()
    os.makedirs(dest, exist_ok=True)
    extract_ = partial(extract_wo_init, model, cfg, skip_completed=False, dest_dir=dest)
    for fpath in tqdm(fpaths):
        rep, path = extract_(fpath)
        np.save(path, rep)
        

def wavlm_init():
    
    # extract wavlm
    # load the pre-trained checkpoints
    device = torch.device("cuda:0")
    if os.path.exists('./wavlm/WavLM-Large.pt'):
        checkpoint = torch.load('./wavlm/WavLM-Large.pt')
    else:
        checkpoint = torch.load('./data/wavlm/WavLM-Large.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg

def extract_wo_init(model, cfg, fpath, skip_completed=True, dest_dir=""):
    device = torch.device("cuda:0")
    # extract the representation of last layer
    wav_input_16khz, _ = librosa.load(fpath, sr=16000)
    wav_input_16khz = torch.from_numpy(wav_input_16khz).to(device)
    wav_input_16khz = wav_input_16khz.unsqueeze(0)
    
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    wavlm_feats = model.extract_features(wav_input_16khz)[0]
    wavlm_feats = wavlm_feats.detach().cpu()
    
    last_feature = wavlm_feats[:, -1, :].unsqueeze(1)
    wavlm_feats = torch.cat((wavlm_feats, last_feature), dim=1)
    
    wavlm_feats = F.interpolate(wavlm_feats.transpose(1, 2), size=math.ceil(wavlm_feats.shape[1] / 2), align_corners=True,
                            mode='linear').transpose(1, 2).squeeze(0)
    wavlm_feats = wavlm_feats.numpy()
    
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return
    return wavlm_feats, save_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")

    args = parser.parse_args()

    extract_folder(args.src, args.dest)
