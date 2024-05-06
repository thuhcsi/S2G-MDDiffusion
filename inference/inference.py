import glob
import os
from tqdm import tqdm
from functools import cmp_to_key
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from LMDM import LMDM
from args import parse_test_opt
from data.slice import slice_audio
from rf_tps.rf_tps_interface import RfTpsInterface
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.wavlm_features import wavlm_init
from data.audio_extraction.wavlm_features import extract_wo_init as wavlm_extract
from skimage import io, img_as_float32
import subprocess
from scipy.interpolate import CubicSpline


# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0
stringintkey = cmp_to_key(stringintcmp_)

def find_best_slice(slice_candidates, last_half):
    last_pos = last_half[-5:] # 5,C
    last_v = last_half[1:] - last_half[:-1] # 39,C
    last_v = np.mean(last_v[-5:], axis=0).reshape(-1,2) # 5,C -> C -> 100,2
    
    min_score = 1000000000
    best_cand = None
    for idx, cand in enumerate(slice_candidates):
        cand_half = cand[:] 
        cand_pos = cand_half[:5] 
        cand_v = cand_half[1:] - cand_half[:-1]
        cand_v = np.mean(cand_v[-5:], axis=0).reshape(-1,2) 
        
        def v_angle_score(array1, array2): # 100,2
            dot_products = np.sum(array1 * array2, axis=1)  
            norms = np.linalg.norm(array1, axis=1) * np.linalg.norm(array2, axis=1)
            cosine_similarity = dot_products / norms
            cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
            angles = np.arccos(cosine_similarity)
            return np.mean(angles)
            
        pos_score = np.sum(np.abs(cand_pos - last_pos))
        v_score = v_angle_score(cand_v*1000, last_v*1000)
        
        final_score = pos_score + v_score

        if final_score < min_score:
            min_score = final_score
            best_cand = cand
    return best_cand


def test(opt):
    # step 1: slice audio
    # step 2: extract audio feature
    # step 3: extract motion feature for the init frame
    # step 3: generate motion feature autoregressively
    # step 4: generate video
    model = LMDM(opt.feature_type, opt.motion_diffusion_ckpt)
    model.eval()
    tps_interface = RfTpsInterface(vis_dir = opt.vis_dir,
                                    config_path = "rf_tps/config/stage3.yaml", ckpt_path = opt.motion_decoupling_ckpt)
    wavlm_model, wavlm_cfg = wavlm_init()
    # step 1: slice audio
    wav_file = opt.wav_file
    
    audio_name = os.path.splitext(os.path.basename(wav_file))[0]
    save_dir = os.path.join(opt.feature_cache_dir, audio_name)
    
    if not os.path.isdir(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(f"Slicing {wav_file}")
        slice_audio(wav_file, 3.2, 3.2, save_dir) 
    print('slice done')
    file_list = sorted(glob.glob(f"{save_dir}/*.wav"), key=stringintkey)
    cond_list = []

    # step 2: extract audio feature
    for idx, file in enumerate(tqdm(file_list)):
        print('extract features')
        wavlm_feats, _ = wavlm_extract(wavlm_model, wavlm_cfg, file)
        baseline_feats, _ = baseline_extract(file)
        cond_list.append(np.concatenate((wavlm_feats, baseline_feats), axis=1))
    cond_list = torch.from_numpy(np.array(cond_list))

    tps_result = []
    
    # step 3: extract motion feature for the init frame
    init_frame = opt.init_frame
    source_image = np.array(img_as_float32(io.imread(init_frame)), dtype='float32')
    init_feature = tps_interface.extract_init_feature(source_image)
    
    # step 4: generate motion features
    for index, cond in enumerate(cond_list):
        if index == 0:
            last_frame = init_feature
            slice_result = model.render_sample(cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal')
            slice_result = slice_result.squeeze().cpu().numpy()
        else:
            last_frame = tps_result[-1][79]
            
            if opt.use_motion_selection:
            # candidates for motion selection
                slice_candidates = []
                for i in range(5):
                    slice_candidate = model.render_sample(cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal')
                    slice_candidate = slice_candidate.squeeze().cpu().numpy()
                    slice_candidates.append(slice_candidate)
                slice_result = find_best_slice(slice_candidates, tps_result[-1])
            else:
                slice_result = model.render_sample(cond_frame=torch.from_numpy(last_frame).float(), cond=cond, last_half=None, mode='normal')
                slice_result = slice_result.squeeze().cpu().numpy()
            
        tps_result.append(slice_result)

    tps_concat_result = tps_result[0]
    for i in range(1, len(tps_result)):
        next_result = tps_result[i]
        tps_concat_result = np.concatenate((tps_concat_result, next_result), axis=0) # T, 200
    
    tps_origin = tps_concat_result
    tps_smoothed = tps_origin.copy()
    smooth_method = 'interpolate' 
                
    if smooth_method == 'interpolate' and opt.use_retrival:
        T = tps_origin.shape[0]
        C = tps_origin.shape[1]
        mutation_points = np.arange(80, T, 80) 
        
        for point in mutation_points:
            start_idx = max(0, point - 5)
            end_idx = min(T, point + 5)
            
            x = list(np.arange(start_idx-3, start_idx)) + list(np.arange(end_idx, end_idx+3))
            y = tps_smoothed[x]
            cs = CubicSpline(x, y, axis=0)  
            xx = np.arange(start_idx-2, end_idx+2)
            interpolated_values = cs(xx)  
            tps_smoothed[start_idx-2:end_idx+2] = interpolated_values
            
    torch.cuda.empty_cache()

    # step 4: generate video
    source_image_path = init_frame
    source_image = np.array(img_as_float32(io.imread(source_image_path)), dtype='float32')
    fname = source_image_path.split('/')[-1].replace(".png", "")
    predictions, refined_predictions, merged_predictions, visualizations, rf_visualizations, mg_visualizations\
        = tps_interface.tps2image(tps_smoothed, source_image)
    video_path = tps_interface.vis_and_write(predictions, refined_predictions, merged_predictions, 
                                                                visualizations, rf_visualizations, mg_visualizations, fname)
    # merge audio and video
    audio_path = wav_file
    video_audio_path = video_path.replace('.mp4', '_audio.mp4')
    subprocess.call(['ffmpeg', 
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    '-shortest', 
                    '-y',
                    video_audio_path])

        
if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
