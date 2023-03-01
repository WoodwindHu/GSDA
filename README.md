# GSDA
<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/GSDA-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-MIT-green">
</a>


GSDA is a point cloud attack in the graph spectral domain

Copyright (C) 2020 Qianjiang Hu, Daizong Liu, Wei Hu

License: MIT for academic use.

Contact: Wei Hu (forhuwei@pku.edu.cn)

## Introduction
With the maturity of depth sensors, point clouds have received increasing attention in various applications such as autonomous driving, robotics, surveillance, \etc., while deep point cloud learning models have shown to be vulnerable to adversarial attacks.
Existing attack methods generally add/delete points or perform point-wise perturbation over point clouds to generate adversarial examples in the data space, which may neglect the geometric characteristics of point clouds.
Instead, we propose point cloud attacks from a new perspective---Graph Spectral Domain Attack (GSDA), aiming to perturb transform coefficients in the graph spectral domain that corresponds to varying certain geometric structure. 
In particular, we naturally represent a point cloud over a graph, and adaptively transform the coordinates of points into the graph spectral domain via graph Fourier transform (GFT) for compact representation. 
We then analyze the influence of different spectral bands on the geometric structure of the point cloud, based on which we propose to perturb the GFT coefficients in a learnable manner guided by an energy constraint loss function. 
Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT (IGFT).
Experimental results demonstrate the effectiveness of the proposed GSDA in terms of both imperceptibility and attack success rates under a variety of defense strategies.


## Requirements
* A computer running on Linux
* NVIDIA GPU and NCCL
* Python 3.6 or higher version
* Pytorch 1.1 or higher version


## Usage

### Model tranining and data preparing
Use `python main.py` to train a new model. Here is an example settings for PointNet:
```
python main_train.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNet --epochs 200
```
Note that `/data/modelnet40_normal_resampled/` is the path of your ModelNet40 dataset. We use the dataset (ModelNet40) of [PointNet++](https://github.com/charlesq34/pointnet2) which can be download [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).


Before running the attack, you can scale down the data and involves only the instances you want to attack, here is an example:
```
python Provider/gen_data_mat.py --datadir /data/modelnet40_normal_resampled/ --npoint 1024 --arch PointNet --out_datadir Data/ --out_classes 10 --max_out_num 25
```
And then the .mat file with 250 instances from 10 different classes would be generated, of which all are correctly classified.

If you DO NOT want to generate the .mat file yourself, you can download one [here](https://drive.google.com/file/d/1mFsEyvfetQDlA30pHijN3S1wAlhwuemk/view?usp=sharing), for the pretrained network provided in `Pretrained/PointNet/1024/`.


### Attack
Use `python attack.py` to generate adversarial point clouds:
```
python main_attack.py --data_dir_file Data/modelnet10_250instances1024_PointNet.mat --npoint 1024 --arch PointNet \
--attack GSDA --attack_label All --binary_max_steps 10 --iter_max_steps 500 \
--cls_loss_type CE --dis_loss_type CD --dis_loss_weight 1.0 --hd_loss_weight 0.1 --spectral_attack --spectral_offset \
--lr 0.01
```

### Defense
`defense.py` is used for evaluating the defense results on the corresponding adversarial point clouds:
```
python defense.py --datadir Exps/PointNet_npoint1024/All/GSDA_0_BiStep10_IterStep500_Optadam_Lr0.01_Initcons10_CE_CDLoss1.0_HDLoss0.1_SpectralAttack0_1024/Mat \
	--npoint 1024 --arch PointNet \
	--defense_type outliers_fixNum --drop_num 128
```

# Citation

```
@article{hu2022exploring,
  title={Exploring the Devil in Graph Spectral Domain for 3D Point Cloud Attacks},
  author={Hu, Qianjiang and Wang, Xiao and Hu, Wei and Qi, Guo-Jun},
  booktitle={European Conference on Computer Vision 2022},
  year={2022}
}
```
