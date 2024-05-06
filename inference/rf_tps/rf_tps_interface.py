import os
import sys
sys.path.append("./rf_tps")
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.e2e_generator import E2EGenerator
from modules.util import get_norm_layer


import matplotlib
matplotlib.use('Agg')
import yaml
import torch
import numpy as np
from logger import Visualizer
import imageio
from skimage import io, img_as_float32


class RfTpsInterface(torch.nn.Module):
    
    def __init__(self, vis_dir,
                 config_path, 
                 ckpt_path
                 ):
        super(RfTpsInterface, self).__init__()
        
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)
            
        self.vis_dir = vis_dir
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        self.inpainting_network = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

        self.kp_detector = KPDetector(**config['model_params']['common_params'])
        
        self.dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
        e2e_generator_params = config['model_params']['e2e_generator_params']
        norm_layer = get_norm_layer(norm_type=e2e_generator_params['norm_type'])
        self.e2e_generator = E2EGenerator(input_nc=e2e_generator_params['input_nc'], output_nc=e2e_generator_params['output_nc'],
                                   ngf=e2e_generator_params['ngf'], norm_layer=norm_layer)
    
        checkpoint = torch.load(ckpt_path)
        self.inpainting_network.load_state_dict(checkpoint['inpainting_network'])
        self.kp_detector.load_state_dict(checkpoint['kp_detector'])
        self.dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        self.e2e_generator.load_state_dict(checkpoint['e2e_generator'])
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.inpainting_network.to(self.device)
        self.kp_detector.to(self.device)
        self.dense_motion_network.to(self.device)
        self.e2e_generator.to(self.device)
        
        self.inpainting_network.eval()
        self.kp_detector.eval()
        self.dense_motion_network.eval()
        self.e2e_generator.eval()
        
        self.num_tps = config['model_params']['common_params']['num_tps']
    
    def tps2tensor(self, tps_feature): # L,200 np
        num_tps = self.num_tps
        kp = tps_feature.reshape(-1, num_tps*5, 2) # L,100,2
        kp = torch.from_numpy(kp).unsqueeze(0).float() # 1,L,100,2 tensor
        
        return kp
    
    def tps2image(self, tps_feature, source_image):
        config = self.config
        
        kp = self.tps2tensor(tps_feature)
        kp = kp.to(self.device)
        source_image = torch.from_numpy(source_image.transpose(2,0,1)).unsqueeze(0).to(self.device) # 1,3,256,256
        
        with torch.no_grad():
            predictions = []
            refined_predictions = []
            merged_predictions = []
            visualizations = []
            rf_visualizations = []
            mg_visualizations = []
            
            kp_source = self.kp_detector(source_image)
            
            for frame_idx in range(kp.shape[1]):
                kp_driving = {'fg_kp': kp[:,frame_idx]}
                dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param = None, 
                                                    dropout_flag = False)
                out = self.inpainting_network(source_image, dense_motion)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                
                merged_images1 = out['prediction']  
                masks = out['occlusion_map'][-1] 
                output_images2 = self.e2e_generator(torch.cat((merged_images1, 1 - masks), dim=1)) 
                merged_images2 = merged_images1 * masks + output_images2 * (1 - masks) 

                out['e2e_prediction'] = output_images2
                out['e2e_merged_prediction'] = merged_images2

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                refined_predictions.append(np.transpose(out['e2e_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                merged_predictions.append(np.transpose(out['e2e_merged_prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                
                visualizer = Visualizer(**config['visualizer_params'])
                visualization = visualizer.visualize_pred(out=out)
                visualizations.append(visualization)
                rf_visualization = visualizer.visualize_rf(out=out)
                rf_visualizations.append(rf_visualization)
                mg_visualization = visualizer.visualize_mg(out=out)
                mg_visualizations.append(mg_visualization)
            
            return predictions, refined_predictions, merged_predictions, visualizations, rf_visualizations, mg_visualizations
        
    def vis_and_write(self, predictions, refined_predictions, merged_predictions, 
                      visualizations, rf_visualizations, mg_visualizations, fname):
        print('save videos...')
        video_path = os.path.join(self.vis_dir, fname+'.mp4')
        imageio.mimsave(video_path, mg_visualizations, fps=self.config["save_fps"], quality=8)
        print('vis done!')
        
        return video_path

    def extract_init_feature(self, source_image):
        source_image = torch.from_numpy(source_image.transpose(2,0,1)).unsqueeze(0).to(self.device)
        kp_source = self.kp_detector(source_image)
        init_feature = kp_source['fg_kp'].squeeze().detach().cpu().numpy().reshape(-1)
        return init_feature

    
