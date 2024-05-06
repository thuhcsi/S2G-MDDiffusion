from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import GeneratorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_
import torch.distributed
from frames_dataset import DatasetRepeater
import math
from tqdm import tqdm

from accelerate import Accelerator

def train(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    # accelerate
    # torch.distributed.init_process_group(backend='gloo', init_method='env://') # for RTX 4090
    accelerator = Accelerator()
    
    train_params = config['train_params']
    optimizer = torch.optim.Adam(
        [{'params': list(inpainting_network.parameters()) +
                    list(dense_motion_network.parameters()) +
                    list(kp_detector.parameters()), 'initial_lr': train_params['lr_generator']}],lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)
    
    optimizer_bg_predictor = None
    if bg_predictor:
        optimizer_bg_predictor = torch.optim.Adam(
            [{'params':bg_predictor.parameters(),'initial_lr': train_params['lr_generator']}], 
            lr=train_params['lr_generator'], betas=(0.5, 0.999), weight_decay = 1e-4)

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint, inpainting_network = inpainting_network, dense_motion_network = dense_motion_network,       
            kp_detector = kp_detector, bg_predictor = bg_predictor,
            optimizer = optimizer, optimizer_bg_predictor = optimizer_bg_predictor)
        print('load success:', start_epoch)
        start_epoch += 1
    else:
        start_epoch = 0

    scheduler_optimizer = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    if bg_predictor:
        scheduler_bg_predictor = MultiStepLR(optimizer_bg_predictor, train_params['epoch_milestones'],
                                              gamma=0.1, last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, 
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, bg_predictor, dense_motion_network, inpainting_network, train_params)
    
    # accelerate preparation
    device = accelerator.device
    generator_full.to(device)
    
    if bg_predictor:
        generator_full, optimizer, dataloader, scheduler_optimizer, optimizer_bg_predictor, scheduler_bg_predictor = accelerator.prepare(
        generator_full, optimizer, dataloader, scheduler_optimizer, optimizer_bg_predictor, scheduler_bg_predictor
    )
    else:   
        generator_full, optimizer, dataloader, scheduler_optimizer = accelerator.prepare(
            generator_full, optimizer, dataloader, scheduler_optimizer
        )
    
    bg_start = train_params['bg_start']
    
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], 
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs'], disable=not accelerator.is_main_process):            
            for x in tqdm(dataloader, disable=not accelerator.is_main_process):
                losses_generator, generated = generator_full(x, epoch)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                
                accelerator.backward(loss)

                clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type = math.inf)
                clip_grad_norm_(dense_motion_network.parameters(), max_norm=10, norm_type = math.inf)
                if bg_predictor and epoch>=bg_start:
                    clip_grad_norm_(bg_predictor.parameters(), max_norm=10, norm_type = math.inf)
                
                optimizer.step()
                optimizer.zero_grad()
                
                if bg_predictor and epoch>=bg_start:
                    optimizer_bg_predictor.step()
                    optimizer_bg_predictor.zero_grad()
                
                
                if accelerator.is_main_process:
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    losses['total'] = loss.detach().data.cpu().numpy()
                    logger.log_iter(losses=losses)

            scheduler_optimizer.step()
            if bg_predictor:
                scheduler_bg_predictor.step()
            
            # save ckpt
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model_unwrapped = accelerator.unwrap_model(generator_full)
                model_save = {
                    'inpainting_network': model_unwrapped.inpainting_network,
                    'dense_motion_network': model_unwrapped.dense_motion_network,
                    'kp_detector': model_unwrapped.kp_extractor,
                    'optimizer': optimizer,
                }
                if bg_predictor and epoch>=bg_start:
                    model_save['bg_predictor'] = model_unwrapped.bg_predictor
                    model_save['optimizer_bg_predictor'] = optimizer_bg_predictor
            
                logger.log_epoch(epoch, model_save, inp=x, out=generated)

