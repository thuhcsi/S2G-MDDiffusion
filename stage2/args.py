import argparse

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name to save ckpts") 
    parser.add_argument("--exp_name", default="exp", help="save to project/name") 
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path") 
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    ) 
    parser.add_argument("--feature_type", type=str, default="wavlm", help="'baseline' or 'wavlm'") 
                    # baseline = hand-crafted features
                    # wavlm = hand-crafted + wavlm features
    parser.add_argument(
        "--wandb_pj_name", type=str, default="LMDM", help="wandb project name"
    ) 
    parser.add_argument("--batch_size", type=int, default=64, help="batch size") # 256
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    ) 
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    ) 
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    ) 
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    ) 
    opt = parser.parse_args()
    return opt


