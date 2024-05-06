<p align="center">

  <h2 align="center">[CVPR'24] Co-Speech Gesture Video Generation via Motion-Decoupled Diffusion Model </h2>
  <p align="center">
    <strong>Xu He</strong></a><sup>1</sup>
    · 
    <strong>Qiaochu Huang</strong></a><sup>1</sup>
    · 
    <strong>Zhensong Zhang</strong></a><sup>2</sup>
    ·
    <strong>Zhiwei Lin</strong></a><sup>1</sup>
    ·
    <strong>Zhiyong Wu</strong></a><sup>1,4</sup>
    ·
    <br><strong>Sicheng Yang</strong></a><sup>1</sup>
    ·  
    ><strong>Minglei Li</strong></a><sup>3</sup>
    ·
    <strong>Zhiyi Chen</strong></a><sup>3</sup>
    ·
    <strong>Songcen Xu</strong></a><sup>2</sup>
    ·
    <strong>Xiaofei Wu</strong></a><sup>2</sup>
    ·
    <br>
    <sup>1</sup>Shenzhen International Graduate School, Tsinghua University  &nbsp;&nbsp;&nbsp; <sup>2</sup>Huawei Noah’s Ark Lab
    <br>
    <sup>3</sup>Huawei Cloud Computing Technologies Co., Ltd   &nbsp;&nbsp;&nbsp; <sup>4</sup>The Chinese University of Hong Kong
    <br>
    </br>
        <a href="https://arxiv.org/abs/2404.01862">
        <img src='https://img.shields.io/badge/arXiv-red' alt='Paper Arxiv'></a> &nbsp; &nbsp; 
        <a href='https://thuhcsi.github.io/S2G-MDDiffusion/'>
        <img src='https://img.shields.io/badge/Project_Page-green' alt='Project Page'></a> &nbsp;&nbsp;
        <!-- <a href='https://www.youtube.com/watch?v=mI8RJ_f3Csw'> -->
        <img src='https://img.shields.io/badge/YouTube-blue' alt='Youtube'></a>
  </p>
    </p>
<div align="center">
  <img src="./assets/teaser.png" alt="Co-Speech Gesture Video Generation via Motion-Decoupled Diffusion Model"></a>
</div>

## 📣 News
* **[2024.05.06]** Release training and inference code with instructions to preprocess the [PATS](https://chahuja.com/pats/download.html) dataset.

* **[2024.03.25]** Release paper.

## 🗒 TODOs
- [x] Release data preprocessing code.
- [x] Release inference code.
- [x] Release pretrained weights.
- [x] Release training code.
- [ ] Release code about evaluation metrics.
- [ ] Release the presentation video.

## ⚒️ Environment
We recommend a python version ```>=3.7``` and cuda version ```=11.7```. It's possible to have other compatible version.

```bash
conda create -n MDD python=3.7
conda activate MDD
pip install -r requirements.txt
```

We test our code on NVIDIA A10, NVIDIA A100, NVIDIA GeForce RTX 4090.

## ⭕ Quick Start
Download our trained weights including ```motion_decoupling.pth.tar``` and ```motion_diffusion.pt``` from [Baidu Netdisk](https://pan.baidu.com/s/1hApZn-MUxofx_pDLeQ34zA?pwd=vdbd). Put them in the ```inference/ckpt``` folder.

Download [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and put it into the ```inference/data/wavlm``` folder.

Now, get started with the following code:

```bash
cd inference
CUDA_VISIBLE_DEVICES=0 python inference.py --wav_file ./assets/001.wav --init_frame ./assets/001.png --use_motion_selection
```

## 📊 Data Preparation
Due to copyright considerations, we are unable to directly provide the preprocessed data subset mentioned in our paper. Instead, we provide the filtered interval ids and preparation instructions. 

To get started, please download the meta file ```cmu_intervals_df.csv``` provided by [PATS](https://chahuja.com/pats/download.html) (you can fint it in any zip file) and put it in the ```data-preparation``` folder. Then run the following code to prepare the data.

```bash
cd data-preparation
bash prepare_data.sh
```
After running the above code, you will get the following folder structure containing the preprocessed data:

```bash
|--- data-preparation
|    |--- data
|    |    |--- img
|    |    |    |--- train
|    |    |    |    |--- chemistry#99999.mp4
|    |    |    |    |--- oliver#88888.mp4
|    |    |    |--- test
|    |    |    |    |--- jon#77777.mp4
|    |    |    |    |--- seth#66666.mp4
|    |    |--- audio
|    |    |    |--- chemistry#99999.wav
|    |    |    |--- oliver#88888.wav
|    |    |    |--- jon#77777.wav
|    |    |    |--- seth#66666.wav
```

## 🔥 Train Your Own Model
Here we use [accelerate](https://github.com/huggingface/accelerate) for distributed training.

### Train the Motion Decoupling Module
Change into the ```stage1``` folder:

```bash
cd stage1
```

Then run the following code to train the motion decoupling module:

```bash 
accelerate launch run.py --config config/pats-256.yaml --mode train
```

Checkpoints be saved in the ```log``` folder, denoted as ```stage1.pth.tar```, which will be used to extract the keypoint features:

```bash
CUDA_VISIBLE_DEVICES=0 python run_extraction.py --config config/pats-256.yaml --mode extraction --checkpoint log/stage1.pth.tar --device_ids 0 --train
CUDA_VISIBLE_DEVICES=0 python run_extraction.py --config config/pats-256.yaml --mode extraction --checkpoint log/stage1.pth.tar --device_ids 0 --test
```

And the extracted motion features will save in the ```feature``` folder.

### Train the Latent Motion Diffusion Module
Change into the ```stage2``` folder:

```bash
cd ../stage2
```

Download [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and put it into the ```data/wavlm``` folder.
Then slice and preprocess the data:

```bash
cd data 
python create_dataset_gesture.py --stride 0.4 --length 3.2 --keypoint_folder ../stage1/feature ----wav_folder ../data-preparation/data/audio --extract-baseline --extract-wavlm
cd ..
```

Run the following code to train the latent motion diffusion module:

```bash
accelerate launch train.py
```

### Training the Refinement Network
Change into the ```stage3``` folder:

```bash
cd ../stage3
```

Download ```mobile_sam.pt``` provided by [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and put it in the ```pretrained_weights``` folder. Then extract bounding boxes of hands for weighted loss (only training set needed):
  
```bash
python get_bbox.py --img_dir ../data-preparation/data/img/train
```

Now you can train the refinement network:

```bash
accelerate launch run.py --config config/stage3.yaml --mode train --tps_checkpoint ../stage1/log/stage1.pth.tar
```

## Citing
If you find our work useful, please consider citing:
```BibTeX
@article{he2024co,
  title={Co-Speech Gesture Video Generation via Motion-Decoupled Diffusion Model},
  author={He, Xu and Huang, Qiaochu and Zhang, Zhensong and Lin, Zhiwei and Wu, Zhiyong and Yang, Sicheng and Li, Minglei and Chen, Zhiyi and Xu, Songcen and Wu, Xiaofei},
  journal={arXiv preprint arXiv:2404.01862},
  year={2024}
}
```

## Acknowledgments

Our code follows several excellent repositories. We appreciate them for making their codes available to the public.
* [Thin-Plate Spline Motion Model for Image Animation](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model)
* [EDGE: Editable Dance Generation From Music](https://github.com/Stanford-TML/EDGE)
* [Image Inpainting with Local and
              Global Refinement](https://github.com/weizequan/LGNet)

