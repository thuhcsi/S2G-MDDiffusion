import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=50, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.epoch_loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        
        self.iter = 0
        self.epoch_count = 0
        
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'run'))

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)
        
    def visualize_rec_multi_frames(self, inp, out):
        image = self.visualizer.visualize_multi_frames(inp['driving'], inp['source_1'], inp['source_2'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, inpainting_network=None, dense_motion_network =None, kp_detector=None, 
                bg_predictor=None, avd_network=None, optimizer=None, optimizer_bg_predictor=None,
                optimizer_avd=None):
        checkpoint = torch.load(checkpoint_path)
        if inpainting_network is not None:
            inpainting_network.load_state_dict(checkpoint['inpainting_network'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if bg_predictor is not None and 'bg_predictor' in checkpoint:
            bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        if dense_motion_network is not None:
            dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        if avd_network is not None:
            if 'avd_network' in checkpoint:
                avd_network.load_state_dict(checkpoint['avd_network'])
        if optimizer_bg_predictor is not None and 'optimizer_bg_predictor' in checkpoint:
            optimizer_bg_predictor.load_state_dict(checkpoint['optimizer_bg_predictor'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if optimizer_avd is not None:
            if 'optimizer_avd' in checkpoint:
                optimizer_avd.load_state_dict(checkpoint['optimizer_avd'])
        epoch = -1
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        return epoch
    
    
    @staticmethod
    def load_cpk_multi_frames(checkpoint_path, inpainting_encoder=None, inpainting_decoder=None, dense_motion_network =None, kp_detector=None, 
                bg_predictor=None, avd_network=None, optimizer=None, optimizer_bg_predictor=None,
                optimizer_avd=None):
        checkpoint = torch.load(checkpoint_path)
        if 'inpainting_network' in checkpoint.keys():   # 微调训练 
            if inpainting_encoder is not None:
                for param_name, param in checkpoint['inpainting_network'].items():
                    if param_name in inpainting_encoder.state_dict().keys():
                        inpainting_encoder.state_dict()[param_name].copy_(param)
                        
            if inpainting_decoder is not None:
                for param_name, param in checkpoint['inpainting_network'].items():
                    if param_name in inpainting_decoder.state_dict().keys():
                        inpainting_decoder.state_dict()[param_name].copy_(param)
                        
        # 测试时
        elif 'inpainting_encoder' in checkpoint.keys() and 'inpainting_decoder' in checkpoint.keys():
            if inpainting_encoder is not None:
                inpainting_encoder.load_state_dict(checkpoint['inpainting_encoder'])
            if inpainting_decoder is not None:
                inpainting_decoder.load_state_dict(checkpoint['inpainting_decoder'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if bg_predictor is not None and 'bg_predictor' in checkpoint:
            bg_predictor.load_state_dict(checkpoint['bg_predictor'])
        if dense_motion_network is not None:
            dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])

        # if optimizer is not None and 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])

        epoch = -1
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        return epoch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    
    def log_iter_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        
        for name, value in zip(loss_names, loss_mean):
            self.writer.add_scalar(name, value, self.iter)
        
        loss_string = 'Epoch:'+str(self.epoch).zfill(self.zfill_num) + "Iter:" + str(self.iter).zfill(self.zfill_num) +')'+ loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()
    
    # def log_iter(self, losses):
    #     losses = collections.OrderedDict(losses.items())
    #     self.names = list(losses.keys())
    #     self.loss_list.append(list(losses.values()))
    
    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        # 每个iter把loss写入iter的loss和epoch的list，每次score iter清空iter的loss，但是直到epoch跑完才清空epoch的loss
        self.loss_list.append(list(losses.values()))
        self.epoch_loss_list.append(list(losses.values()))
        self.log_iter_scores(self.names)
        
        self.iter += 1

    def log_epoch_scores(self, loss_names):
        loss_mean = np.array(self.epoch_loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = 'Epoch total:'+ str(self.epoch).zfill(self.zfill_num) + ") " + loss_string + '\n'
        
        for name, value in zip(loss_names, loss_mean):
            name = 'epoch_'+name
            self.writer.add_scalar(name, value, self.epoch_count)

        print(loss_string, file=self.log_file)
        self.epoch_loss_list = []
        self.log_file.flush()
    
    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_epoch_scores(self.names)
        self.visualize_rec(inp, out)
        self.epoch_count += 1

    def log_epoch_multi_frames(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_epoch_scores(self.names)
        self.visualize_rec_multi_frames(inp, out)
        self.epoch_count += 1
    
    
    
def draw_colored_heatmap(heatmap, colormap, bg_color):
    parts = []
    weights = []
    bg_color = np.array(bg_color).reshape((1, 1, 1, 3))
    num_regions = heatmap.shape[-1]
    for i in range(num_regions):
        color = np.array(colormap(i / num_regions))[:3]
        color = color.reshape((1, 1, 1, 3))
        part = heatmap[:, :, :, i:(i + 1)]
        part = part / np.max(part, axis=(1, 2), keepdims=True)
        weights.append(part)

        color_part = part * color
        parts.append(color_part)

    weight = sum(weights)
    bg_weight = 1 - np.minimum(1, weight)
    weight = np.maximum(1, weight)
    result = sum(parts) / weight + bg_weight * bg_color
    return result

class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2 # 100,2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        source = source.data.cpu()
        kp_source = out['kp_source']['fg_kp'].data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source, kp_source))

        # # Equivariance visualization
        # if 'transformed_frame' in out:
        #     transformed = out['transformed_frame'].data.cpu().numpy()
        #     transformed = np.transpose(transformed, [0, 2, 3, 1])
        #     transformed_kp = out['transformed_kp']['fg_kp'].data.cpu().numpy()
        #     images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))
        
        # # Driving image with mask
        # if 'hand_mask' in out:
        #     hand_mask = out['hand_mask'].unsqueeze(3).data.cpu().repeat(1, 1, 1, 3).numpy()
        #     driving = driving * (1 - hand_mask)
        #     images.append(driving)

        # Deformed image
        if 'deformed' in out:
            deformed = out['deformed'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['fg_kp'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)


        ## Occlusion map
        if 'occlusion_map' in out:
            for i in range(len(out['occlusion_map'])):
                occlusion_map = out['occlusion_map'][i].data.cpu().repeat(1, 3, 1, 1)
                occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
                occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
                images.append(occlusion_map)
                
        # if 'hand_mask' in out:
        #     hand_mask = out['hand_mask'].unsqueeze(1).data.cpu().repeat(1, 3, 1, 1)
        #     hand_mask = np.transpose(hand_mask, [0, 2, 3, 1])
        #     images.append(hand_mask)

        # Deformed images according to each individual transform
        # if 'deformed_source' in out:
        #     full_mask = []
        #     for i in range(out['deformed_source'].shape[1]):
        #         image = out['deformed_source'][:, i].data.cpu()
        #         # import ipdb;ipdb.set_trace()
        #         image = F.interpolate(image, size=source.shape[1:3])
        #         mask = out['contribution_maps'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
        #         mask = F.interpolate(mask, size=source.shape[1:3])
        #         image = np.transpose(image.numpy(), (0, 2, 3, 1))
        #         mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

        #         if i != 0:
        #             color = np.array(self.colormap((i - 1) / (out['deformed_source'].shape[1] - 1)))[:3]
        #         else:
        #             color = np.array((0, 0, 0))

        #         color = color.reshape((1, 1, 1, 3))

        #         images.append(image)
        #         if i != 0:
        #             images.append(mask * color)
        #         else:
        #             images.append(mask)

        #         full_mask.append(mask * color)

        #     images.append(sum(full_mask))
        
        
        # # Heatmaps visualizations
        # if 'heatmap' in out['kp_driving']:
        #     driving_heatmap = F.interpolate(out['kp_driving']['heatmap'], size=source.shape[1:3])
        #     driving_heatmap = np.transpose(driving_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
        #     images.append(draw_colored_heatmap(driving_heatmap, self.colormap, self.region_bg_color))

        # if 'heatmap' in out['kp_source']:
        #     source_heatmap = F.interpolate(out['kp_source']['heatmap'], size=source.shape[1:3])
        #     source_heatmap = np.transpose(source_heatmap.data.cpu().numpy(), [0, 2, 3, 1])
        #     images.append(draw_colored_heatmap(source_heatmap, self.colormap, self.region_bg_color))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
    
    
    def visualize_simple(self, driving, source, out):
        images = []

        # # Source image with keypoints
        # source = source.data.cpu()
        # kp_source = out['kp_source']['fg_kp'].data.cpu().numpy()
        # source = np.transpose(source, [0, 2, 3, 1])
        # images.append((source, kp_source))

        # # Driving image with keypoints
        # kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        # driving = driving.data.cpu().numpy()
        # driving = np.transpose(driving, [0, 2, 3, 1])
        # images.append((driving, kp_driving))

        # # Deformed image
        # if 'deformed' in out:
        #     deformed = out['deformed'].data.cpu().numpy()
        #     deformed = np.transpose(deformed, [0, 2, 3, 1])
        #     images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['fg_kp'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)


        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image



    def visualize_multi_frames(self, driving, source_1, source_2, out):
        images = []

        # Source image with keypoints
        source_1 = source_1.data.cpu()
        kp_source_1 = out['kp_source_1']['fg_kp'].data.cpu().numpy()
        source_1 = np.transpose(source_1, [0, 2, 3, 1])
        images.append((source_1, kp_source_1))

        source_2 = source_2.data.cpu()
        kp_source_2 = out['kp_source_2']['fg_kp'].data.cpu().numpy()
        source_2 = np.transpose(source_2, [0, 2, 3, 1])
        images.append((source_2, kp_source_2))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Deformed image 两个soure分别warp及融合后的结果
        if 'deformed_1' in out:
            deformed_1 = out['deformed_1'].data.cpu().numpy()
            deformed_1 = np.transpose(deformed_1, [0, 2, 3, 1])
            images.append(deformed_1)

        if 'deformed_2' in out:
            deformed_2 = out['deformed_2'].data.cpu().numpy()
            deformed_2 = np.transpose(deformed_2, [0, 2, 3, 1])
            images.append(deformed_2)
            
        if 'deformed_source' in out:
            deformed = out['deformed_source'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)
        
        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['fg_kp'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)
        
        if 'occlusion_last' in out:
            occlusion_last = out['occlusion_last'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_last = np.transpose(occlusion_last, [0, 2, 3, 1])
            images.append(occlusion_last)
        


        ## Occlusion map
        if 'occlusion_map' in out:
            for i in range(len(out['occlusion_map'])):
                occlusion_map = out['occlusion_map'][i].data.cpu().repeat(1, 3, 1, 1)
                occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
                occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
                images.append(occlusion_map)

        # Deformed images according to each individual transform
        # if 'deformed_source' in out:
        #     full_mask = []
        #     for i in range(out['deformed_source'].shape[1]):
        #         image = out['deformed_source'][:, i].data.cpu()
        #         # import ipdb;ipdb.set_trace()
        #         image = F.interpolate(image, size=source.shape[1:3])
        #         mask = out['contribution_maps'][:, i:(i+1)].data.cpu().repeat(1, 3, 1, 1)
        #         mask = F.interpolate(mask, size=source.shape[1:3])
        #         image = np.transpose(image.numpy(), (0, 2, 3, 1))
        #         mask = np.transpose(mask.numpy(), (0, 2, 3, 1))

        #         if i != 0:
        #             color = np.array(self.colormap((i - 1) / (out['deformed_source'].shape[1] - 1)))[:3]
        #         else:
        #             color = np.array((0, 0, 0))

        #         color = color.reshape((1, 1, 1, 3))

        #         images.append(image)
        #         if i != 0:
        #             images.append(mask * color)
        #         else:
        #             images.append(mask)

        #         full_mask.append(mask * color)

        #     images.append(sum(full_mask))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
    
    def visualize_multi_frames_simple(self, driving, source_1, source_2, out):
        images = []

        # Source image with keypoints
        source_1 = source_1.data.cpu()
        kp_source_1 = out['kp_source_1']['fg_kp'].data.cpu().numpy()
        source_1 = np.transpose(source_1, [0, 2, 3, 1])
        images.append((source_1, kp_source_1))

        source_2 = source_2.data.cpu()
        kp_source_2 = out['kp_source_2']['fg_kp'].data.cpu().numpy()
        source_2 = np.transpose(source_2, [0, 2, 3, 1])
        images.append((source_2, kp_source_2))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        # Result with and without keypoints
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        if 'kp_norm' in out:
            kp_norm = out['kp_norm']['fg_kp'].data.cpu().numpy()
            images.append((prediction, kp_norm))
        images.append(prediction)
        

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image



class VisualizerTest:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2 # 5,2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                # 按照五个一组划分，每组对应tps的五个点
                for tps_ind in range(arg[1].shape[1]//10):
                    kp = arg[1][:, tps_ind*10:(tps_ind+1)*10]
                    out.append(self.create_image_column_with_kp(arg[0], kp))
                
                # # 画需要的点
                # kp = arg[1][:, [22,23,61,73]]
                # out.append(self.create_image_column_with_kp(arg[0], kp))
            else:
                out.append(self.create_image_column(arg))
        images = []
        for row in range(2):
            out_row = out[row*5:(row+1)*5]
            images.append(np.concatenate(out_row, axis=1)) # 256,256*5,3
        return np.concatenate(images, axis=0) # 256*4,256*5,3
    
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        #Driving image with keypoints
        kp_driving = out['kp_driving']['fg_kp'].data.cpu().numpy()
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving, kp_driving))

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image



    