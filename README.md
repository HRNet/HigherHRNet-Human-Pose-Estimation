# [Higher-Resolution Networks (HigherHRNet) for Human Pose Estimation](https://arxiv.org/abs/1908.10357)

## News
* \[2019/11/23\] Code and models for [HigherHRNet](https://arxiv.org/abs/1908.10357) are now released!
* \[2019/08/27\] HigherHRNet is now on [ArXiv](https://arxiv.org/abs/1908.10357). We will also release code and models, stay tuned!

## Introduction
This is the official code of [Bottom-up Higher-Resolution Networks for Multi-Person Pose Estimation](https://arxiv.org/abs/1908.10357).  
Bottom-up multi-person pose estimation methods have difficulties in predicting the correct pose for small persons due to challenges in scale variation. In this paper, we present a Higher-Resolution Network (HigherHRNet) to learn high-resolution feature pyramids. Equipped with mutli-resolution supervision for training and multi-resolution aggregation  for inference, the proposed approach is able to solve the scale variation challenge in *bottom-up multi-person* pose estimation and localize the keypoints, especially for small person, more precisely. The feature pyramid in HigherHRNet consists of the feature map output from HRNet and the upsampled higher-resolution one through a transposed convolution. HigherHRNet outperforms the previous best bottom-up method by 2.5% AP for medium person on COCO test-dev, showing its effectiveness in handling scale variation. Furthermore, HigherHRNet achieves a state-of-the-art result of 70.5% AP on COCO test-dev without using refinement or other post-processing techniques, surpassing all existing bottom-up methods.

![Illustrating the architecture of the proposed Higher-HRNet](/figures/arch_v2.png)

## Main Results
### Results on COCO val2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 67.1  | 86.2  |  73.0  |  61.5  |  76.1  | 
| HigherHRNet        | HRNet-w32  | 640      |  28.6M  | 74.8   | 68.5  | 87.1  |  74.7  |  64.3  |  75.3  | 
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 69.9  | 87.2  |  76.1  |  65.4  |  76.4  | 

### Results on COCO val2017 *with* multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 69.9  | 87.1  |  76.0  |  65.3  |  77.0  | 
| HigherHRNet        | HRNet-w32  | 640      |  28.6M  | 74.8   | 70.6  | 88.1  |  76.9  |  66.6  |  76.5  | 
| HigherHRNet        | HRNet-w48  | 640      |  63.8M  | 154.3  | 72.1  | 88.4  |  78.2  |  67.8  |  78.3  | 

### Results on COCO test-dev2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| OpenPose\*         |    -     | -          |   -     |  -     | 61.8  | 84.9  |  67.5  |  57.1  |  68.2  | 
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 56.6  | 81.8  |  61.8  |  49.8  |  67.0  | 
| PersonLab          | ResNet-152  | 1401    |  68.7M  | 405.5  | 66.5  | 88.0  |  72.6  |  62.4  |  72.3  |
| PifPaf             |    -     | -          |   -     |  -     | 66.7  | -     |  -     |  62.4  |  72.9  | 
| Bottom-up HRNet    | HRNet-w32  | 512      |  28.5M  | 38.9   | 64.1  | 86.3  |  70.4  |  57.4  |  73.9  | 
| **HigherHRNet**    | HRNet-w32  | 512      |  28.6M  | 47.9   | 66.4  | 87.5  |  72.8  |  61.2  |  74.2  | 
| **HigherHRNet**    | HRNet-w48  | 640      |  63.8M  | 154.3  | **68.4**  | **88.2**  |  **75.1**  |  **64.4**  |  **74.2**  | 

### Results on COCO test-dev2017 *with* multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 63.0  | 85.7  |  68.9  |  58.0  |  70.4  | 
| Hourglass\*        | Hourglass  | 512      | 277.8M  | 206.9  | 65.5  | 86.8  |  72.3  |  60.6  |  72.6  | 
| PersonLab          | ResNet-152  | 1401    |  68.7M  | 405.5  | 68.7  | 89.0  |  75.4  |  64.1  |  75.5  | 
| **HigherHRNet**    | HRNet-w48  | 640      |  63.8M  | 154.3  | **70.5**  | **89.3**  |  **77.2**  |  **66.6**  |  **75.8**  | 

*Note: \* indicates using refinement.*

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA P100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
5. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/open?id=1bdXVmYrSynPLSk5lptvgyQ8fhziobD50) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW4AwKRMklXVzndJT0))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   `-- hrnet_w32-36af842e.pth
            `-- pose_coco
                `-- pose_higher_hrnet_w32_512.pth

   ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- crowd_pose
    `-- |-- json
        |   |-- crowdpose_train.json
        |   |-- crowdpose_val.json
        |   `-- crowdpose_test.json
        `-- images
            |-- 100000.jpg
            |-- 100001.jpg
            |-- 100002.jpg
            |-- 100003.jpg
            |-- 100004.jpg
            |-- 100005.jpg
            |-- ... 
```

### Training and Testing

#### Testing on COCO val2017 dataset using model zoo's models ([GoogleDrive](https://drive.google.com/drive/folders/1X9-TzWpwbX2zQf2To8lB-ZQHMYviYYh6?usp=sharing))
 

For single-scale testing:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth
```

By default, we use horizontal flip. To test without flip:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.FLIP_TEST False
```

Multi-scale testing is also supported, although we do not report results in our paper:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth \
    TEST.SCALE_FACTOR '[0.5, 1.0, 2.0]'
```


#### Training on COCO train2017 dataset

```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

By default, it will use all available GPUs on the machine for training. To specify GPUs, use

```
CUDA_VISIBLE_DEVICES=0,1 python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

#### Mixed-precision training
Due to large input size for bottom-up methods, we use mixed-precision training to train our Higher-HRNet by using the following command:
```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    FP16.ENABLED True FP16.DYNAMIC_LOSS_SCALE True
```

Our code for mixed-precision training is borrowed from [NVIDIA Apex API](https://github.com/NVIDIA/apex).


### Other applications
Many other dense prediction tasks, such as segmentation, face alignment and object detection, etc. have been benefited by HRNet. More information can be found at [Deep High-Resolution Representation Learning](https://jingdongwang2017.github.io/Projects/HRNet/).

## Citation
If you find this work or code is helpful in your research, please cite:
````
@article{cheng2019bottom,
  title={Bottom-up Higher-Resolution Networks for Multi-Person Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  journal   = {CoRR},
  volume    = {abs/1908.10357},
  year={2019}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={arXiv preprint arXiv:1908.07919},
  year={2019}
}
````

