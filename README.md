# AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition
AMS-Net: Modeling Adaptive Multi-granularity
Spatio-temporal Cues for Video Action Recognition

This is an code implementation of AMS-Net[[paper]](), created by Qilong Wang, Qiyao Hu and Zilin Gao.
## Introduction

Effective spatio-temporal modeling as a core of
video representation learning is challenged by complex scale
variations of spatio-temporal cues in videos, especially different visual tempos of actions and varying spatial sizes of moving objects. Most of existing works handle complex spatio-temporal scale variations based on input-level or feature-level pyramid mechanisms, which however, rely on expensive multi-stream architectures or fuse coarse-level spatio-temporal features only in a fixed manner. To effectively capture complex scale dynamics of spatio-temporal cues in an efficient way, this paper proposes a single-stream architecture with single-input (namely AMS-Net) to model Adaptive Multi-granularity Spatio-temporal cues for video action recognition. To this end, our AMS-Net proposes two
core components (i.e., competitive progressive temporal modeling (CPTM) block and collaborative spatio-temporal pyramid
(CSTP) module) to respectively capture fine-grained temporal
cues and fuse coarse-level spatio-temporal features in an adaptive manner, aiming to handle subtle variations of visual tempos and fair-sized spatio-temporal dynamics in a unified architecture. Our AMS-Net can be flexibly instantiated based on existing deep convolutional neural networks (CNNs) with the proposed CPTM block and CSTP module. The experiments are conducted on eight video benchmarks, and the results show our AMSNet establishs state-of-the-art performance on fine-grained action recognition (i.e., Diving48 and FineGym), while performing very competitively on widely used Something-Something and Kinetics.

## Citation
```
@inproceedings{wang2022ams_net,
      title={AMS-Net: Modeling Adaptive Multi-granularity
Spatio-temporal Cues for Video Action Recognition},
      author={Wang, Qilong and Hu, Qiyao and Gao, Zilin and Li, Peihua and Hu, Qinghua},
      booktitle={TNNLS},
      year={2023}
    }
```

## Overview - AMS Net

![AMS—Net_arch](fig/AMS-Net-overview.jpg)

## Our Environment and Configuration
```
OS: Ubuntu 16.04
CUDA: 10.2/11.1
Toolkit: PyTorch 1.7
GPU: GTX 2080Ti/TiTan RTX
```

## Installation
```shell
conda create -n ams_net python=3.8 -y
conda activate ams_net
conda install pytorch torchvision -c pytorch

# install mmaction package
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
pip install mmaction2

# install ams-net package
git clone https://github.com/ParkHUQ/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition.git
cd AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition
pip install -v -e .
```

## Usage
You can use the following command to train AMS-Net.
```
sh train_AMS_R50_sthv1_rgb_8f.sh
```
or
```
python tools/train.py configs/recognition/ams/ams_r50_1x1x8_110e_sthv1_rgb.py 8  --work-dir work_dirs/ams_r50_1x1x8_110e_sthv1_rgb --seed 0 --deterministic
```
You can use the following command to test a model
```
python tools/test.py configs/recognition/ams/ams_r50_1x1x8_110e_sthv1_rgb.py \
    model_hub/SomethingV1/ams_sthv1_8fx1x1.pth --eval top_k_accuracy mean_class_accuracy --average-clips prob --dump result.pkl
```

## Model Zoo

### Something-Something V1/V2
|Method|Backbone|Pertrain|Frames|1-crop(%)|multi-view(%)|Model|Config|
|:----:|:------:|:------:|:----:|:-------:|:---------:|:----:|:----:|
|AMS-Net|2D ResNet50|ImageNet-1K|8F|50.4/80.2|53.2/81.7|[SSV1_AMS_TSN_R50_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x8_110e_sthv1_rgb.py)|
|AMS-Net*|2D ResNet50|ImageNet-1K|8F|53.1/82.3|54.2/82.6|[SS_SSV1_AMS_TSN_R50_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ss_ams_r50_1x1x8_110e_sthv1_rgb.py)|
|AMS-Net|2D ResNet50|ImageNet-1K|16F|52.4/81.7|54.2/82.5|[SSV1_AMS_TSN_R50_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x16_110e_sthv1_rgb.py)|
|AMS-Net*|2D ResNet50|ImageNet-1K|16F|53.6/82.4|54.4/82.6|[SS_SSV1_AMS_TSN_R50_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ss_ams_r50_1x1x16_110e_sthv1_rgb.py)|
|AMS-Net(E)|2D ResNet50|ImageNet-1K|8F+16F|55.3/83.6|56.1/83.7|--|--|

|Method|Backbone|Pertrain|Frames|1-crop(%)|multi-view(%)|Model|Config|
|:----:|:------:|:------:|:----:|:-------:|:---------:|:----:|:----:|
|AMS-Net|2D ResNet50|ImageNet-1K|8F|62.4/88.6|65.3/89.7|[SSV2_AMS_TSN_R50_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x8_120e_sthv2_rgb.py)|
|AMS-Net*|2D ResNet50|ImageNet-1K|8F|65.4/90.2|66.7/90.8|[SS_SSV2_AMS_TSN_R50_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ss_ams_r50_1x1x8_120e_sthv2_rgb.py)|
|AMS-Net|2D ResNet50|ImageNet-1K|16F|64.1/89.8|66.0/90.5|[SSV2_AMS_TSN_R50_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x16_120e_sthv2_rgb.py)|
|AMS-Net*|2D ResNet50|ImageNet-1K|16F|65.6/90.3|66.9/90.9|[SS_SSV2_AMS_TSN_R50_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ss_ams_r50_1x1x16_120e_sthv2_rgb.py)|
|AMS-Net(E)|2D ResNet50|ImageNet-1K|8F+16F|67.0/90.9|67.8/91.4|--|--|

### FineGym & Diving48
- FineGym 99 & FineGym 288

|Method|Dataset|Backbone|Pertrain|Frames|Mean(%)|Top-1(%)|Model|Config|
|:----:|:-------:|:------:|:------:|:----:|:-------:|:---------:|:----:|:----:|
|AMS-Net|Gym99|SlownOnly R50|ImageNet-1K|16F|91.9|94.3|[AMS_SlowOnlyR50_Gym99_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_slowonlyR50_pretrained_16x4x1_110e_gym99_rgb.py)|
|AMS-Net|Gym288|SlownOnly R50|ImageNet-1K|16F|65.3|90.9|[AMS_SlowOnlyR50_Gym288_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_slowonlyR50_pretrained_r50_16x4x1_110e_gym288_rgb.py)|

- Diving 48

|Method|Dataset Version|Backbone|Pertrain|Frames|Top-1(%)|Model|Config|
|:----:|:-------:|:-----:|:------:|:----:|:-------:|:------:|:----:|
|AMS-Net|V1|2D ResNet50|ImageNet-1K|16F|44.4|[AMS_Diving48_V1_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x16_75e_diving48_rgb.py)|
|AMS-Net|V2|2D ResNet50|ImageNet-1K|16F|90.1|[AMS_Diving48_V2_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_r50_1x1x16_75e_diving48_rgb.py)|

### Kinetics 400

|Method|Dataset|Backbone|Pertrain|Frames|Top-1(%)|Top-5(%)|Model|Config|
|:----:|:-------:|:------:|:------:|:----:|:-------:|:---------:|:----:|:----:|
|AMS-Net|mini-K200|SlownOnly R50|ImageNet-1K|8F|81.7|-|[AMS_SlowOnlyR50_MiniK200_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_slowonlyR50_imagenet_pretrained_8x8x1_60e_kinetics200_rgb.py)|
|AMS-Net|K400|SlownOnly R50|ImageNet-1K|8F|77.1|93.0|[AMS_SlowOnlyR50_Kinetics400_8f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_slowonlyR50_imagenet_pretrained_8x8x1_130e_kinetics400_rgb.py)|
|AMS-Net|K400|SlownOnly R50|ImageNet-1K|16F|78.0|93.3|[AMS_SlowOnlyR50_Kinetics400_16f]()|[Config](/AMS-Net-Adaptive-Multi-granularity-Spatio-temporal-Network-for-Video-Action-Recognition/configs/recognition/ams/ams_slowonlyR50_imagenet_pretrained_16x4x1_130e_kinetics400_rgb.py)|


## Code components
```
AMS-Net
├── mmaction
|    ├── models
|    |   ├── backbones
|    |   |      ├── ams_2D_module.py
|    |   |      ├── ams_3D_module.py
|    |   |      ├── ams_resnet3d.py
|    |   |      ├── ams_resnet3d_slowfast.py
|    |   |      ├── ams_resnet3d_slowonly.py
|    |   |      └── snippet_sample_resnet_ams.py        
|    |   ├──  necks
|    |   |      └──cstp.py
|    |   ├──  heads
|    |   |      └──ams_head.py
|    ├ ...
├── configs
|   ├── recoginition
|   |      ├── ams
|   |      |    ├── ams_r50_1x1x8_110e_sthv1_rgb.py
|   |      |    ├──ams_r50_1x1x8_120e_sthv2_rgb.py
|   |      |    ├ ...
├ ...
├── data
|   ├── somthtingv1
|   |      ├── train_videofolder.txt
|   |      └── val_videofolder.txt  
|   ├ ...
├ ...
├── scripts
|   ├── train_AMS_R50_sthv1_rgb_8f.sh
|   ├──train_AMS_R50_sthv1_rgb_16f
|   ├ ... 
├ ...
```

## Acknowledgments
- We really appreciate the works as well as the accompanying code of [TSN](https://github.com/yjxiong/tsn-pytorch), [TSM](https://github.com/mit-han-lab/temporal-shift-module), [SlowFast](https://github.com/facebookresearch/SlowFast), [TPN](https://github.com/decisionforce/TPN) and [MMaction2](https://github.com/open-mmlab/mmaction2) toolbox. Thank for all hard research work and dedication.

## Contact Information
If you have any questions or suggestions, please feel free to leave a message or contact us directly via email: <huqy@tju.edu.cn>; <qlwang@tju.edu.cn>.
