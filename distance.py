import argparse
import os
import os.path
import json
import numpy as np
import sys
from pytorch3d.ops import knn_points, knn_gather
# import provider
from scipy.io import loadmat
import torch
from Lib.loss_utils import *
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def load_pointcloud(file_path):
    point_set = np.loadtxt(file_path,delimiter=',').astype(np.float32)
    # Take the first npoints
    point_set = point_set[0:1024,:]
    point_set[:,0:3] = pc_normalize(point_set[:,0:3])
    point_set = point_set[:,0:3]
    return point_set

def array2samples_distance(array1, array2):
    num_point, num_features = array1.shape
    expanded_array1 = np.tile(array1, (num_point, 1))
    expanded_array2 = np.reshape(
            np.tile(np.expand_dims(array2, 1),
                    (1, num_point, 1)),
            (-1, num_features))
    distances = np.linalg.norm(expanded_array1-expanded_array2, axis=1)
    distances = np.reshape(distances, (num_point, num_point))
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances
 
def chamfer_distance_numpy(array1, array2):
    num_point, num_features = array1.shape
    dist = 0
    av_dist1 = array2samples_distance(array1, array2)
    av_dist2 = array2samples_distance(array2, array1)
    dist = dist + (av_dist1+av_dist2)
    return dist


parser = argparse.ArgumentParser(description='Point Cloud Attacking')
parser.add_argument('--ours_data_path', type=str, default='')
args = parser.parse_args()
ours_data_path = args.ours_data_path
gt_data_path = './Data/gt'

ours_cd_distance_numpy = 0.0
ours_cd_distance_torch = 0.0
ours_hd_distance_torch = 0.0
ours_l2_distance_torch = 0.0
ours_num = 0
file_list = os.listdir(ours_data_path)
with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn()) as progress:
    file_tqdm = progress.add_task(description="file progress", total=len(file_list))
    for file in file_list:
        progress.advance(file_tqdm, advance=1)
        if 'factor' in file:
            continue
        file_path = os.path.join(ours_data_path, file)
        file_gt_path = os.path.join(gt_data_path, file.split('_')[1]+'.mat')
        ours_cd_distance_numpy += chamfer_distance_numpy(loadmat(file_gt_path)['adversary_point_clouds'].transpose(1,0), loadmat(file_path)['adversary_point_clouds'].transpose(1,0))
        ours_cd_distance_torch += chamfer_loss(torch.from_numpy(loadmat(file_gt_path)['adversary_point_clouds']).unsqueeze(0), torch.from_numpy(loadmat(file_path)['adversary_point_clouds']).unsqueeze(0))
        ours_hd_distance_torch += hausdorff_loss(torch.from_numpy(loadmat(file_gt_path)['adversary_point_clouds']).unsqueeze(0), torch.from_numpy(loadmat(file_path)['adversary_point_clouds']).unsqueeze(0))
        ours_l2_distance_torch += norm_l2_loss(torch.from_numpy(loadmat(file_gt_path)['adversary_point_clouds']).unsqueeze(0), torch.from_numpy(loadmat(file_path)['adversary_point_clouds']).unsqueeze(0))
        ours_num += 1

print('ours_cd_distance_numpy: ', ours_cd_distance_numpy/ours_num)
print('ours_cd_distance_torch: ', ours_cd_distance_torch/ours_num)
print('ours_hd_distance_torch: ', ours_hd_distance_torch/ours_num)
print('ours_l2_distance_torch: ', ours_l2_distance_torch/ours_num)