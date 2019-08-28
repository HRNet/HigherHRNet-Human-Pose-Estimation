# [Higher-Resolution Networks (HigherHRNet) for Human Pose Estimation](https://arxiv.org/abs/1908.10357)

## News
* \[2019/08/27\] HigherHRNet is now on [ArXiv](https://arxiv.org/abs/1908.10357). We will also release code and models, stay tuned!

## Introduction
This is the official code of [Bottom-up Higher-Resolution Networks for Multi-Person Pose Estimation](https://arxiv.org/abs/1908.10357).  
In this work, we are interested in bottom-up multi-person human pose estimation. A typical bottom-up pipeline consists of two main steps: heatmap prediction and keypoint grouping. We mainly focus on the first step for improving heatmap prediction accuracy. We propose Higher-Resolution Network (HigherHRNet), which is a simple extension of the High-Resolution Network (HRNet). HigherHRNet generates higher-resolution feature maps by deconvolving the high-resolution feature maps outputted by HRNet, which are spatially more accurate for small and medium persons. Then, we build high-quality multi-level features and perform multi-scale pose prediction. The extra computation overhead is marginal and negligible in comparison to existing bottom-up methods that rely on multi-scale image pyramids or large input image size to generate accurate pose heatmaps. HigherHRNet surpasses all existing bottom-up methods on the COCO dataset without using multi-scale test. 

## Main Results
### Results on COCO test-dev2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| OpenPose\*         |    -     | -          |   -     |  -     | 61.8  | 84.9  |  67.5  |  57.1  |  68.2  | 
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 56.6  | 81.8  |  61.8  |  49.8  |  67.0  | 
| PersonLab          | ResNet-101  | 1401    |  68.7M  | 405.5  | 66.5  | 88.0  |  72.6  |  62.4  |  72.3  | 
| Bottom-up HRNet    | HRNet-w32  | 512      |  28.5M  | 38.9   | 64.1  | 86.3  |  70.4  |  57.4  |  73.9  | 
| **HigherHRNet**    | HRNet-w32  | 512      |  28.6M  | 44.6   | 66.4  | 87.5  |  72.8  |  61.2  |  74.2  | 
| **HigherHRNet**    | HRNet-w48  | 640      |  63.8M  | 154.3  | **68.4**  | **88.2**  |  **75.1**  |  **64.4**  |  **74.2**  | 

### Results on COCO test-dev2017 with multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 63.0  | 85.7  |  68.9  |  58.0  |  70.4  | 
| Hourglass\*        | Hourglass  | 512      | 277.8M  | 206.9  | 65.5  | 86.8  |  72.3  |  60.6  |  72.6  | 
| PersonLab          | ResNet-101  | 1401    |  68.7M  | 405.5  | 68.7  | 89.0  |  75.4  |  64.1  |  75.5  | 
| **HigherHRNet**    | HRNet-w48  | 640      |  63.8M  | 154.3  | **70.5**  | **89.3**  |  **77.2**  |  **66.6**  |  **75.8**  | 

*Note: \* indicates using refinement.*

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
