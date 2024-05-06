import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        if start_idx == 0:
            start_idx += stride_step
        else:
            audio_slice = audio[start_idx : start_idx + window]
            sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
            start_idx += stride_step
            idx += 1
    return idx


def slice_keypoint(keypoint_file, stride, length, num_slices, out_dir):
    keypoint = np.load(keypoint_file)
    file_name = os.path.splitext(os.path.basename(keypoint_file))[0]
    start_idx = 0
    window = int(length * 25)
    stride_step = int(stride * 25)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(keypoint) - window and slice_count < num_slices:
        if start_idx == 0:
            start_idx += stride_step
        else:
            # save the first frame as condition
            keypoint_slice = keypoint[start_idx - 1 : start_idx + window]
            np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", keypoint_slice)
            start_idx += stride_step
            slice_count += 1
    return slice_count

def slice_wavlm(wavlm_file, stride, length, num_slices, out_dir):
    # stride, length in seconds
    wavlm = np.load(wavlm_file)
    file_name = os.path.splitext(os.path.basename(wavlm_file))[0]
    start_idx = 0
    slice_count = 0
    window = int(length * 25)
    stride_step = int(stride * 25)
    while start_idx <= len(wavlm) - window and slice_count < num_slices:
        if start_idx == 0:
            start_idx += stride_step
        else:
            wavlm_slice = wavlm[start_idx : start_idx + window]
            np.save(f"{out_dir}/{file_name}_slice{slice_count}.npy", wavlm_slice)
            start_idx += stride_step
            slice_count += 1
    return slice_count
        
def slice_data(keypoint_dir, wav_dir, wavlm_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    keypoints = sorted(glob.glob(f"{keypoint_dir}/*.npy"))
    wavlm = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"
    keypoint_out = keypoint_dir + "_sliced"
    wavlm_out = wavlm_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(keypoint_out, exist_ok=True)
    os.makedirs(wavlm_out, exist_ok=True)
    print(len(wavs))
    print(len(keypoints))
    print(len(wavlm))
    assert len(wavs) == len(keypoints) == len(wavlm)
    for wav, keypoint, wavlm in tqdm(zip(wavs, keypoints, wavlm)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(keypoint))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        wavlm_name = os.path.splitext(os.path.basename(wavlm))[0]
        assert m_name == w_name == wavlm_name, str((keypoint, wav, wavlm_name))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        keypoint_slices = slice_keypoint(keypoint, stride, length, audio_slices, keypoint_out)
        wavlm_slices = slice_wavlm(wavlm, stride, length, audio_slices, wavlm_out)
        # make sure the slices line up
        assert audio_slices == keypoint_slices == wavlm_slices, str(
            (wav, keypoint, wavlm, audio_slices, keypoint_slices, wavlm_slices)
        )

def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
        
def slice_wavlm_folder(wavlm_dir, stride, length):
    wavlms = sorted(glob.glob(f"{wavlm_dir}/*.npy"))
    wavlm_out = wavlm_dir[:-4]
    os.makedirs(wavlm_out, exist_ok=True)
    for wavlm in tqdm(wavlms):
        slice_wavlm(wavlm, stride, length, wavlm_out)
