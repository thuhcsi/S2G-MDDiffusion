from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.pipeline_E2E import PipelineModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
from frames_dataset import DatasetRepeater
import math
from tqdm import tqdm
from functools import partial

from accelerate import Accelerator

import numpy as np
import torch.distributed


def train(config, inpainting_network, kp_detector, dense_motion_network, 
          e2e_generator, gan_discriminator, tps_checkpoint, log_dir, dataset):
    
    # torch.distributed.init_process_group(backend='gloo', init_method='env://') # for RTX 4090
    
    # accelerate
    accelerator = Accelerator()
    
    train_params = config['train_params']
    
    optimizer_e2e = torch.optim.Adam(e2e_generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)
    optimizer_gan = torch.optim.Adam(gan_discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    start_epoch = 0

    scheduler_e2e = MultiStepLR(optimizer_e2e, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)
    scheduler_gan = MultiStepLR(optimizer_gan, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
 
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, 
                            num_workers=train_params['dataloader_workers'], drop_last=True)
    
    pipeline_model = PipelineModel(accelerator,
                                   kp_detector, dense_motion_network, inpainting_network, 
                                   e2e_generator, gan_discriminator, 
                                   optimizer_e2e, optimizer_gan,
                                   scheduler_e2e, scheduler_gan,
                                   train_params, tps_checkpoint)
    pipeline_model.setup() 
    dataloader = accelerator.prepare(dataloader)
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs'], disable=not accelerator.is_main_process):            
            for x in tqdm(dataloader, disable=not accelerator.is_main_process):
                losses_generator, generated = pipeline_model.optimize_parameters(x) 

                if accelerator.is_main_process:
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)
                
            pipeline_model.update_learning_rate()
            
            
            # Save model
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_tps_model = accelerator.unwrap_model(pipeline_model.tps_model)
                unwrapped_e2e_generator = accelerator.unwrap_model(pipeline_model.e2e_generator)
                unwrapped_gan_discriminator = accelerator.unwrap_model(pipeline_model.gan_discriminator)
                logger.log_epoch(epoch, {'kp_detector': unwrapped_tps_model.kp_extractor,
                                         'dense_motion_network': unwrapped_tps_model.dense_motion_network,
                                         'inpainting_network': unwrapped_tps_model.inpainting_network,
                                         'e2e_generator': unwrapped_e2e_generator,
                                         'gan_discriminator': unwrapped_gan_discriminator,
                                         'optimizer_e2e': optimizer_e2e,
                                         'optimizer_gan': optimizer_gan}, inp=x, out=generated)
