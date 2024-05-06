import argparse

def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="wavlm")
    parser.add_argument("--wav_file", type=str, default="assets/001.wav")
    parser.add_argument("--init_frame", type=str, default="assets/001.png") 
    parser.add_argument('--use_motion_selection', action='store_true', help='use motion selection')
    parser.add_argument(
        "--motion_diffusion_ckpt", type=str, default="ckpt/motion_diffusion.pt", help="motion diffusion checkpoint"
    )
    parser.add_argument(
        "--motion_decoupling_ckpt", type=str, default="ckpt/motion_decoupling.pth.tar", help="motion decoupling checkpoint"
    )

    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features",
        help="Where to cache the features",
    ) 
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="visulization",
        help="Where to save visual results",
    ) 
    opt = parser.parse_args()
    return opt

