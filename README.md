# Learnable Oriented-Derivative Network for Polyp Segmentation

## Abstract

Gastrointestinal polyps are the main cause of colorectal cancer.  Given the polyp variations in terms of size, color, texture and poor optical conditions brought by endoscopy, polyp segmentation is still a challenging problem. In this paper, we propose a Learnable Oriented-Derivative Network (LOD-Net) to refine the accuracy of boundary predictions for polyps. Specifically, it firstly calculates eight oriented derivatives at each pixel for a polyp. It then selects those pixels with large oriented-derivative values to constitute a candidate border region of a polyp. It finally refines boundary prediction by fusing border region features and also those high-level semantic features calculated by a backbone network. Extensive experiments and ablation studies show that the proposed LOD-Net achieves superior performance compared to the state-of-the-art methods by a significant margin on publicly available datasets, including CVC-ClinicDB, Kvasir, ETIS, and EndoScene. 




## Datasets

We follow the data setting of [PraNet](https://github.com/DengPingFan/PraNet). You could download the datasets and process them to COCO-format for training and testing.

## Results
|                     |CVC-ClinicDB  |       Kvasir   |        ETIS    |     EndoScene   |
|---------------------|--------------------|-----------------------------|--------------------|----------|
|Model                |  mDice    /  mIoU   |  mDice   /  mIoU   |  mDice   /  mIoU   |  mDice   /  mIoU   |
| LOD_R_101_FPN_1x    |   92.6   /  86.2   |   93.9   /   88.4  |   93.8   /   88.4  |   95.7   /   91.7  |



## Installation

Our project is developed on [detectron2](https://github.com/facebookresearch/detectron2). Please follow the official detectron2 [installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). All our code is under `projects/LOD-Net/`. 

##Training
The train command line is same with detectron2:
'''
cd /path/to/detectron2/projects/LOD-Net/
python train_net.py --config-file configs/LOD_R_101_FPN_1x.yaml --num-gpus 4
'''

##Testing
You can use "--eval-only" parameter for testing.
'''
cd /path/to/detectron2/projects/LOD-Net/
python train_net.py --config-file configs/LOD_R_101_FPN_1x.yaml --eval-only MODEL.WEIGHTS outputs/LOD_R_101_FPN_1x/model_final.pth
'''
