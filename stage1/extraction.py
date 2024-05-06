import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger
import json


def extraction(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, dataset, feature_dir, is_train):

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, kp_detector=kp_detector,
                         bg_predictor=bg_predictor, dense_motion_network=dense_motion_network)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if is_train:
        feature_dir = os.path.join(feature_dir, 'train')
    else:
        feature_dir = os.path.join(feature_dir, 'test')
    
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir, exist_ok=True)
    
    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        print("Extracting features for video: ", x['name'][0])
        
        with torch.no_grad():
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
                
            kp_list = []

            for frame_idx in range(x['video'].shape[2]): # bs,c,t,h,w
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving) # 1,100,2
                kp_list.append(kp_driving['fg_kp'].squeeze().cpu().tolist()) # T,100,2
            
            feature_path = os.path.join(feature_dir, x['name'][0]+'.json')
            
            with open(feature_path, 'w') as f:
                json.dump({'kp':kp_list}, f)

    print("Extraction done!")
