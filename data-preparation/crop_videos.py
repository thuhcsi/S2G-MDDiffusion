import imageio
import os
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
from skimage.transform import resize
import json
from skimage.util import img_as_ubyte
import subprocess
warnings.filterwarnings("ignore")

def get_video(input_path, video_path, left_edge_ratio, img_shape, speaker):
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)
    else:
        print("【Already exists】",input_path.split('/')[-1])
        return
    reader = imageio.get_reader(input_path)
    frames = []
    try:
        for i, frame in enumerate(reader):
            # no longer than 15s
            if i > 374:
                break
            hight = frame.shape[0]
            width = frame.shape[1]
            len_need = hight
            left_edge = int(width * left_edge_ratio)
            crop = frame[:, left_edge:left_edge+len_need]

            if speaker == 'jon':
                left_edge = width - len_need
                crop = frame[:, left_edge:]
            assert crop.shape[0] == crop.shape[1], "crop shape is not square"
            crop = img_as_ubyte(resize(crop, img_shape, anti_aliasing=True))
            frames.append(crop)

    except imageio.core.format.CannotReadFrameError:
        None
    
    for idx, frame in enumerate(frames):
        frame_name = f"{idx:07d}"
        frame_name = frame_name + '.png'
        imageio.imsave(os.path.join(video_path, frame_name), frame)

    print('【Succeed】',input_path.split('/')[-1])
    return

def get_audio(input_path, audio_path):
    subprocess.call(['ffmpeg', '-i', input_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-t', '15', audio_path])
    if os.path.exists(audio_path):
        print('【Succeed】',input_path.split('/')[-1])
    else:
        print('【Failed】',input_path.split('/')[-1])
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trim_video_dir", default='trim_data', help='path to trimed videos')
    parser.add_argument("--speaker", default='oliver', help="speaker name")
    parser.add_argument("--filter_path", default="filtered_intervals.json", help='path to the filtered intervals json file')
    parser.add_argument("--img_shape", default=(256,256), help='shape of the output images')
    parser.add_argument("--video_dir", default='data/img', help='path to the processed frame images')
    parser.add_argument("--audio_dir", default='data/audio', help='path to the corresponding audio files')
    
    # hand-crafted left edge ratio for each speaker
    left_edge_radios = {'seth': 0.395, 'oliver': 0.38, 'chemistry': 0.031, 'jon': 0.43}
    
    args = parser.parse_args()
    speaker = args.speaker
    speaker_dir = os.path.join(args.trim_video_dir, speaker)
    video_all_dir = args.video_dir
    audio_all_dir = args.audio_dir
    img_shape = args.img_shape
    
    # for split info
    with open(args.filter_path, "r") as json_file:
        filter_dict = json.load(json_file)
    train_ids = filter_dict["train"]
    test_ids = filter_dict["test"]

    for split in ['train', 'test']:
        video_dir = os.path.join(video_all_dir, split)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
    if not os.path.exists(audio_all_dir):
        os.makedirs(audio_all_dir, exist_ok=True)
        
    for file in tqdm(sorted(os.listdir(speaker_dir))):
        if file.endswith(".mp4"):
            input_path = os.path.join(speaker_dir, file)
            id = int(file.split('#')[-1].replace(".mp4", ""))
            split = 'train' if id in train_ids else 'test'
            video_path = os.path.join(video_all_dir, split, file.replace(".mp4", "")) # save each video mp4 as a folder containing png frame images
            audio_path = os.path.join(audio_all_dir, file.replace(".mp4", ".wav")) # no need to split corresponding audio files into train/test folders
            if not os.path.exists(video_path):
                get_video(input_path, video_path, left_edge_radios[speaker], img_shape, speaker)
                get_audio(input_path, audio_path)