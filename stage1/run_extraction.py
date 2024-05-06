import matplotlib
matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from frames_dataset import FramesDatasetExtraction
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
import torch
from extraction import extraction
import os 


if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/pats-256.yaml", help="path to config")
    parser.add_argument("--mode", default="extraction", choices=["extraction"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default='ckpt/stage1.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--feature_dir", default='feature', help="path to store features")
    parser.add_argument("--part", default="train", choices=["train","test"], help="extraction for train or test set")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
        
        
    if not os.path.exists(opt.feature_dir):
        os.makedirs(opt.feature_dir, exist_ok=True)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:'+str(opt.device_ids[0]))
        inpainting.to(cuda_device)

    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
                                                           
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        dense_motion_network.to(opt.device_ids[0])

    bg_predictor = None
    avd_network = None

    dataset = FramesDatasetExtraction(is_train=(opt.part.startswith('train')), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'extraction':
        print("Extraction...")
        extraction(config, inpainting, kp_detector, bg_predictor, dense_motion_network, opt.checkpoint, dataset, opt.feature_dir, is_train=(opt.part.startswith('train')))    
    else:
        raise Exception("Mode not supported")