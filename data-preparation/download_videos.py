import argparse
import pandas as pd
import os
from tqdm import tqdm
import json

def downloader(row):
    link = row['video_link']
    if 'youtube' in link:
        try:
            video_path = row["video_link"][-11:] + ".mp4"
            if os.path.exists(video_path):
                print(" 【Exists】" + video_path + '\n') 
                return
            else:
                print(" 【Begin】" + video_path + '\n')
                cmd = "yt-dlp -f 'bv[ext=mp4]+ba[ext=m4a]' --embed-metadata --merge-output-format mp4 " + link + " -o " + video_path
                os.system(cmd)
                if os.path.exists(video_path):
                    print(" 【Succeed】" + video_path + '\n')
                else:
                    print(" 【Failed】" + video_path + '\n')
        except Exception as e:
            print(e)
    else: 
        try:
            video_path = row["video_link"][-4:] + ".mp4"
            if os.path.exists(video_path):
                print(" 【Exists】【Media link】" + video_path + '\n') 
                return
            else:
                print(" 【Begin】【Media link】" + video_path + '\n')
                cmd = "yt-dlp " + link + " -o " + video_path
                os.system(cmd)
                if os.path.exists(video_path):
                    print(" 【Succeed】【Media link】" + video_path + '\n')
                else:
                    print(" 【Failed】【Media link】" + video_path + '\n')
        except Exception as e:
            print(e)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-speaker', '--speaker', default="oliver", help='download videos of a specific speaker')
    parser.add_argument('-meta_path', '--meta_path', default="cmu_intervals_df.csv", help='path to the intervals_df meta file')
    parser.add_argument('-filter_path', '--filter_path', default="filtered_intervals.json", help='path to the filtered intervals json file')
    args = parser.parse_args()

    df = pd.read_csv(args.meta_path)
    if args.speaker:
        df = df[df['speaker'] == args.speaker]
    
    # filter
    with open(args.filter_path, "r") as json_file:
        filter_dict = json.load(json_file)
    train_ids = filter_dict["train"]
    test_ids = filter_dict["test"]
    keep_indices = df["interval_id"].astype(int).isin(train_ids + test_ids)
    df = df[keep_indices]
    
    df_download = df.drop_duplicates(subset=['video_link'])
    for _, row in tqdm(df_download.iterrows(), total=df_download.shape[0]):
        downloader(row)