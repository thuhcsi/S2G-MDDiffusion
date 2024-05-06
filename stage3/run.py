import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from shutil import copy
from frames_dataset import FramesDatasetHandAugment
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.gan import SNDiscriminator
from modules.e2e_generator import E2EGenerator
from modules.util import init_net, get_norm_layer
import torch
from train import train
import random
import numpy as np


if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/stage3.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--tps_checkpoint", default=None, help="path to pre-trained tps checkpoint to restore")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")


    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.safe_load(f)
        
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True 

    if opt.mode == 'train' and opt.tps_checkpoint is not None: 
        log_dir = os.path.join(*os.path.split(opt.tps_checkpoint)[:-1])

    kp_detector = KPDetector(**config['model_params']['common_params'])
    kp_detector = init_net(kp_detector)
    
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    dense_motion_network = init_net(dense_motion_network)
    
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    inpainting = init_net(inpainting)
                                                           
    
    e2e_generator_params = config['model_params']['e2e_generator_params']
    norm_layer = get_norm_layer(norm_type=e2e_generator_params['norm_type'])
    e2e_generator = E2EGenerator(input_nc=e2e_generator_params['input_nc'], output_nc=e2e_generator_params['output_nc'],
                                   ngf=e2e_generator_params['ngf'], norm_layer=norm_layer)
    e2e_generator = init_net(e2e_generator, e2e_generator_params['init_type'], e2e_generator_params['init_gain'])

    
    gan_discriminator_params = config['model_params']['gan_discriminator_params']
    gan_discriminator = SNDiscriminator(in_channels = gan_discriminator_params['in_channels'])
    gan_discriminator = init_net(gan_discriminator, 
                                    gan_discriminator_params['init_type'], gan_discriminator_params['init_gain'])
    


    dataset = FramesDatasetHandAugment(is_train=(opt.mode.startswith('train')), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, inpainting, kp_detector, dense_motion_network, e2e_generator, gan_discriminator, opt.tps_checkpoint, log_dir, dataset)
    else:
        raise ValueError("Mode not supported")
