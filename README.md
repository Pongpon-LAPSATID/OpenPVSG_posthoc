# About This Repo
This repo contains my experiment adding the post-hoc logit adjustment operations [1] to the panoptic video scene graph generation framework [2] to mitigate long-tailed distribution bias in the training set.

This repo is a deliverable for my final assignment in the "4840-1014 Visual Media" coursework (2026 S1/S2 semester), provided by the Graduate School of Information Science and Technology, The University of Tokyo.

**Environment:**
Hardware: CP-VELUGA G3 RTX A5000 (DualBoot) Laptop
OS: Ubuntu20.04LTS
CPU: Core i7 11800H 8core/16thread 2.3GHz (TurboBoost 4.6GHz)
Memory: 32 GB (16GB x2) GPU: NVIDIA RTX A5000 Laptop 16GB GDDR6

**Implementation Steps:**
1. Follow the guidelines from the original PVSG paper [2] below to download the dataset and pre-trained models. 
2. My implementation did not include the videos from "EpicKitchen" dataset. To follow my steps, download the .json files from this URL and put them under the data/ folder: https://drive.google.com/drive/folders/15aHa414H1cllMz2VvYP3xKLKN30vXXhU?usp=sharing
Note: main_results.csv in the URL recorded the results from my implementation. I provided this as a reference only.
3. Run the scripts below. Adjust the relative paths and other variable names in .sh and .py scripts as necessary to run the test scripts.
/ Tracking and save query features
sh scripts/utils/prepare_qf_ips.sh
/ Prepare for relation modeling
sh scripts/utils/prepare_rel_set.sh
/ Train relation models
sh scripts/train/train_relation.sh
/ Test
sh scripts/test/test_relation_full.sh

**References:**
[1] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, “Long-tail learning via logit adjustment,” Oct. 2020. Accessed: July 31, 2026. [Online]. Available: https://openreview.net/forum?id=37nvvqkCo5.<br>
[2] J. Yang et al., "Panoptic Video Scene Graph Generation," 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 18675-18685, doi: 10.1109/CVPR52729.2023.01791.

# [README from Original PVSG Paper] Panoptic Video Scene Graph Generation
<p align="center">
  <!-- <img src="./assets/psgtr_long.gif" align="center" width="80%"> -->

https://github.com/Jingkang50/OpenPVSG/assets/17070708/54a0f4c4-daca-4168-8460-95eb4cf8b85a

<video controls>
  <source src="[https://github.com/Jingkang50/OpenPVSG/assets/17070708/54a0f4c4-daca-4168-8460-95eb4cf8b85a](https://github.com/Jingkang50/OpenPVSG/assets/17070708/54a0f4c4-daca-4168-8460-95eb4cf8b85a)" type="video/mp4">
  Your browser does not support the video tag.
</video>

  <p align="center">
  <a href="https://arxiv.org/abs/2311.17058" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-CVPR%202023-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://jingkang50.github.io/PVSG/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-jingkang50/PVSG-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpHpnXP-ta9Nu1wD6FwkDWAB0LxY8oE9VNqsgv6ln-i8QQ?e=fURefF" target='_blank'>
    <img src="https://img.shields.io/badge/Data-PVSGDataset-334b7f?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgvpTfCTMudLpxw-h0_BVdcBAHacUaAQD-u9OvkUlpaDBg?e=LXnqaX" target='_blank'>
    <img src="https://img.shields.io/badge/Data-QuickView-7de5f6?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://github.com/LilyDaytoy/OpenPVSG" target='_blank'>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLilyDaytoy%2FPVSG&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=true">
  </p>
  </a>
  <p align="center">
  <font size=5><strong>Panoptic Video Scene Graph Generation</strong></font>
    <br>
        <a href="https://jingkang50.github.io/">Jingkang Yang</a>,
        <a href="https://lilydaytoy.github.io/">Wenxuan Peng</a>,
        <a href="https://lxtgh.github.io/">Xiangtai Li</a>,<br>
        <a href="https://scholar.google.com/citations?user=G8DPsoUAAAAJ&amp;hl=zh-CN">Zujin Guo</a>,
        <a href="https://cliangyu.com/"> Liangyu Chen</a>,
        <a href="https://brianboli.com/">Bo Li</a>,
        <a href="https://www.linkedin.com/in/zheng-ma-4201223a/?originalSubdomain=hk">Zheng Ma</a>,<br>
        <a href="https://kaiyangzhou.github.io/">Kaiyang Zhou</a>,
        <a href="https://bmild.github.io/">Wayne Zhang</a>,
        <a href="https://www.mmlab-ntu.com/person/ccloy/">Chen Change Loy</a>,
        <a href="https://liuziwei7.github.io/">Ziwei Liu</a>,
    <br>
  S-Lab, Nanyang Technological University & SenseTime Research
  </p>
</p>

---
## What is PVSG Task?
<strong>The Panoptic Video Scene Graph Generation (PVSG) Task</strong> aims to interpret a complex scene video with a dynamic scene graph representation, with each node in the scene graph grounded by its pixel-accurate segmentation mask tube in the video.

| ![pvsg.jpg](assets/teaser.png) |
|:--:|
| <b>Given a video, PVSG models need to generate a dynamic (temporal) scene graph that is grounded by panoptic mask tubes.</b>|


## The PVSG Dataset
We carefully collect 400 videos, each featuring dynamic scenes and rich in logical reasoning content. On average, these videos are 76.5 seconds long (5 FPS). The collection comprises 289 videos from VidOR, 55 videos from EpicKitchen, and 56 videos from Ego4D.

Please access the dataset via this [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EpHpnXP-ta9Nu1wD6FwkDWAB0LxY8oE9VNqsgv6ln-i8QQ?e=fURefF), and put the downloaded zip files to the place below.
```
├── assets
├── checkpoints
├── configs
├── data
├── data_zip
│   ├── Ego4D
│   │   ├── ego4d_masks.zip
│   │   └── ego4d_videos.zip
│   ├── EpicKitchen
│   │   ├── epic_kitchen_masks.zip
│   │   └── epic_kitchen_videos.zip
│   ├── VidOR
│   │   ├── vidor_masks.zip
│   │   └── vidor_videos.zip
│   └── pvsg.json
├── datasets
├── models
├── scripts
├── tools
├── utils
├── .gitignore
├── environment.yml
└── README.md
```
Please run `unzip_and_extract.py` to unzip the files and extract frames from the videos. If you use `zip`, make sure to use `unzip -j xxx.zip` to remove junk paths. You should have your `data` directory looks like this:
```
data
├── ego4d
│   ├── frames
│   ├── masks
│   └── videos
├── epic_kitchen
│   ├── frames
│   ├── masks
│   └── videos
├── vidor
│   ├── frames
│   ├── masks
│   └── videos
└── pvsg.json
```

We suggest our users to play with `./tools/Visualize_Dataset.ipynb` to quickly get familiar with PSG dataset.

## Get Started
To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
conda activate openpvsg
```
You shall manually install the following dependencies.
```bash
# Install mmcv
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
conda install -c conda-forge pycocotools
pip install mmdet==2.25.0

# already within environment.yml
pip install timm
python -m pip install scipy
pip install git+https://github.com/cocodataset/panopticapi.git

# for unitrack
pip install imageio==2.6.1
pip install lap==0.4.0
pip install cython_bbox==0.1.3

# for vps
pip install seaborn
pip install ftfy
pip install regex

# If you're using wandb for logging
pip install wandb
wandb login
```

Download the [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) for tracking if you are interested in IPS+Tracking solution.


## Training and Testing
**IPS+Tracking & Relation Modeling**
```bash
# Train IPS
sh scripts/train/train_ips.sh
# Tracking and save query features
sh scripts/utils/prepare_qf_ips.sh
# Prepare for relation modeling
sh scripts/utils/prepare_rel_set.sh
# Train relation models
sh scripts/train/train_relation.sh
# Test
sh scripts/test/test_relation_full.sh
```

**VPS & Relation Modeling**
```bash
# Train VPS
sh scripts/train/train_vps.sh
# Save query features
sh scripts/utils/prepare_qf_vps.sh
# Prepare for relation modeling
sh scripts/utils/prepare_rel_set.sh
# Train relation models
sh scripts/train/train_relation.sh
# Test
sh scripts/test/test_relation_full.sh
```
## Model Zoo
Method    | M2F ckpt | vanilla | filter |  conv |  transformer |
---       | ---  | ---  | ---  | ---  | ---  |
mask2former_ips | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) |
mask2former_vps | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) | [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErwH2H27bJpAg9xpaTa49fkB3IJkiLJ6AEFuxUHYKMI1dQ?e=9XINcP) |

## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{yang2023pvsg,
    author = {Yang, Jingkang and Peng, Wenxuan and Li, Xiangtai and Guo, Zujin and Chen, Liangyu and Li, Bo and Ma, Zheng and Zhou, Kaiyang and Zhang, Wayne and Loy, Chen Change and Liu, Ziwei},
    title = {Panoptic Video Scene Graph Generation},
    booktitle = {CVPR},
    year = {2023},
}
```
