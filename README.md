# Learnable Oriented-Derivative Network for Polyp Segmentation

## Abstract

Gastrointestinal polyps are the main cause of colorectal cancer.  Given the polyp variations in terms of size, color, texture and poor optical conditions brought by endoscopy, polyp segmentation is still a challenging problem. In this paper, we propose a Learnable Oriented-Derivative Network (LOD-Net) to refine the accuracy of boundary predictions for polyps. Specifically, it firstly calculates eight oriented derivatives at each pixel for a polyp. It then selects those pixels with large oriented-derivative values to constitute a candidate border region of a polyp. It finally refines boundary prediction by fusing border region features and also those high-level semantic features calculated by a backbone network. Extensive experiments and ablation studies show that the proposed LOD-Net achieves superior performance compared to the state-of-the-art methods by a significant margin on publicly available datasets, including CVC-ClinicDB, Kvasir, ETIS, and EndoScene. 




## Datasets

We follow the data setting of [PraNet](https://github.com/DengPingFan/PraNet). You could download the datasets and process them to COCO-format for training and testing.

## Results
We use the metric code of [mmSegmentation](https://github.com/open-mmlab/mmsegmentation) which is an open source project of OpenMMLab to calculate the mDice and mIoU, shown in seman_mask_evaluate.py. We evaluate the result of previous works based on result maps supported by [PraNet](https://github.com/DengPingFan/PraNet) (U-Net, U-Net++, SFA, PraNet) and our prediction on official model weights (HarDNet). 

|                     |CVC-ClinicDB  |       Kvasir   |        ETIS    |  CVC-ColonDB    |     CVC-T   |
|---------------------|--------------------|-----------------------------|----|----------------|----------|
|Model                |  mDice    /  mIoU   |  mDice   /  mIoU   |  mDice   /  mIoU  |  mDice   /  mIoU  |  mDice   /  mIoU   |
|U-Net++	|78.57/	64.7	|73.54	/58.15|	45.11/	29.12|	31.60/	18.76	|63.96/	47.02|
|SFA	|74.02/	58.75|	66.07/	49.32	|71.79	/56.00|	56.03	/38.92	|84.4/	73.01|
|PraNet	|95.12	/90.69|	90.63	/82.87	|84.2	/72.71|	67.09/	50.48|	95.07	/90.6|
|HarDNet|	95.32	/91.05|	90.51	/82.67|	83.91	/72.29|	58.17	/41.02	|94.44	/89.47|
|                   |                     |                     |                    | |                     |
|Mask R-CNN(baseline)|	92.48	/86.02	|  92.39	/85.87	|  89.74 / 81.39 |	58.64 /	41.48 |	94	/ 88.68        |
| LOD_R_101_FPN_1x    |   92.6   /  86.2   |   93.9   /   88.4  |   93.8   /   88.4  |   70.03	/53.88 |	95.69	/91.73|


## Loss
Total Loss
![image](https://github.com/midsdsy/LOD-Net/blob/master/imgs/total_loss.png)
OD_Loss
![image](https://github.com/midsdsy/LOD-Net/blob/master/imgs/od_loss.png)

## Visualization 
Here the visualization of learned oriented derivatives in instances. We chose several random pixels (too crowd to show all) in instances to show.

## Installation

Our project is developed on [detectron2](https://github.com/facebookresearch/detectron2). Please follow the official detectron2 [installation](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). All our code is under `projects/LOD-Net/`. 

## Training
The train command line is same with detectron2:
```
cd /path/to/projects/LOD-Net/
python train_net.py --config-file configs/LOD_R_101_FPN_1x.yaml --num-gpus 4
```

## Testing
We use seman_mask_generation.py to generate the result map of model. Change the path of result maps in seman_mask_evaluate.py and run.
```
cd /path/to/projects/LOD-Net/
python seman_mask_generation.py
python seman_mask_evaluate.py
```
