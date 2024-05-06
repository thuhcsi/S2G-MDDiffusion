import torch
import cv2
import copy
import numpy as np
import mediapipe as mp
import os
from mobile_sam import sam_model_registry, SamPredictor
import json
from tqdm import tqdm
from argparse import ArgumentParser

# mediapipe for hand detection
min_detection_confidence = 0.7
min_tracking_confidence = 0.5
use_static_image_mode = False
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)    

# box for hand segmentation prompt
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return np.array([x, y, x + w, y + h])

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    joint = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_v = landmark.visibility
        landmark_point.append([landmark_x, landmark_y])
        joint.append([landmark_x, landmark_y, landmark_v])
    return np.array(landmark_point), joint


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir", default=None, help="path to train dataset")
    parser.add_argument("--bbox_dir", default="bbox", help="path to bounding box")

    opt = parser.parse_args()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = opt.train_dir
    bbox_dir = opt.bbox_dir
    
    if not os.path.isdir(bbox_dir):
        os.makedirs(bbox_dir, exist_ok=True)
    
    # load SAM model
    model_type = "vit_t"
    sam_checkpoint = "./pretrained_weights/mobile_sam.pt"
    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
            
    for video_name in tqdm(sorted(os.listdir(img_dir))[4160:]): # video loop
        print(video_name)
        video_box_list = []
        video_path = os.path.join(img_dir, video_name)
        for img_name in tqdm(sorted(os.listdir(video_path))): # frame loop
            img_path = os.path.join(video_path, img_name)
            img = cv2.imread(img_path)
            debug_image = copy.deepcopy(img)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            predictor.set_image(imgRGB)
            label_rgb = np.zeros((256,256,3))
            label_rgb_2 = np.zeros((256,256,3))
            
            frame_box_list = [] # accurate hand box
            
            # hand loop
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                        results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list, joints = calc_landmark_list(debug_image, hand_landmarks)
                    point_labels = [1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1] if len(landmark_list)>0 else []
                    masks, _, _ = predictor.predict(multimask_output=False, point_coords=landmark_list, point_labels=point_labels, box=brect)
                    
                    label_rgb[masks[0]==1] = [255,255,255] 
                    
                    kernel = np.ones((3,3),np.uint8)
                    mask = cv2.morphologyEx(masks[0].astype(np.uint8), cv2.MORPH_OPEN, kernel)

                    label_rgb_2[mask==1] = [255,255,255] 
                    
                    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
                    box = [x, y, x + w, y + h]
                    
                    box_to_save = [y, x, y + h, x + w] 
                    frame_box_list.append(box_to_save)
                    
            video_box_list.append(frame_box_list)
            
        box_path = os.path.join(bbox_dir, video_name.split('.')[0]+'.json')
        with open(box_path, 'w') as f:
            json.dump({'box':video_box_list}, f) # ndim=3, (frame,hands,points), [[],[[1,2,3,4]],[[1,2,3,4][1,2,3,4]]]
            
