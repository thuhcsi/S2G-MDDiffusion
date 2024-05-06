import pandas as pd
import os
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import json
warnings.filterwarnings("ignore")

def crop_video(interval, raw_video_dir, cropped_video_dir):
    try:
        start_time = str(pd.to_datetime(interval['start_time']).time())
        end_time = str(pd.to_datetime(interval['end_time']).time())
        video_id = (interval["video_link"][-11:])

        if (interval["speaker"] == 'jon') and ('youtube' not in interval["video_link"]):
            video_id = (interval["video_link"])[-4:]
        
        input_fn = os.path.join(raw_video_dir, video_id + ".mp4")
        if os.path.exists(input_fn):
            output_fn = os.path.join(cropped_video_dir, "%s#%s.mp4"%(interval['speaker'],interval['interval_id'])) 
            if not(os.path.exists(output_fn)):
                print("\n 【Begin】" + output_fn)
                cmd = 'ffmpeg -i "%s"  -ss %s -to %s -r 25 "%s" -y' % (
                    input_fn, str(start_time), str(end_time), output_fn)
                os.system(cmd)
                
                if not(os.path.exists(output_fn)):
                    print("\n 【Failed】" + output_fn)
                else:
                    print("\n 【Succeed】" + output_fn)
                
        else: 
            print("\n 【No source】" +  video_id +'.mp4')
    except Exception as e:
        print(e)
        print("couldn't crop interval: %s"%interval)
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_video_dir", default='raw_data', help='Path to raw videos')
    parser.add_argument("--metadata", default='cmu_intervals_df.csv', help='Path to metadata')
    parser.add_argument("--trimed_video_dir", default='trim_data', help='Path to output')
    parser.add_argument("--speaker", default='oliver', help="Speaker name")
    parser.add_argument("--filter_path", default="filtered_intervals.json", help='path to the filtered intervals json file')

    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df['speaker'] == args.speaker]
    
    # filter
    with open(args.filter_path, "r") as json_file:
        filter_dict = json.load(json_file)
    train_ids = filter_dict["train"]
    test_ids = filter_dict["test"]
    keep_indices = df["interval_id"].astype(int).isin(train_ids + test_ids)
    df = df[keep_indices]
    
    raw_video_dir = os.path.join(args.raw_video_dir, args.speaker)
    trimed_video_dir = os.path.join(args.trimed_video_dir, args.speaker)
    if not os.path.exists(trimed_video_dir):
        os.makedirs(trimed_video_dir, exist_ok=True)

    for _, interval in tqdm(df.iterrows(), total=df.shape[0]):
        crop_video(interval, raw_video_dir, trimed_video_dir)
        
    print("Done!")