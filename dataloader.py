from common.parser import *

import yaml
import torch
import os
import numpy as np



root='/home/share/dataset/semanticKITTI'
DATA = yaml.safe_load(open('params/semantic-kitti.yaml', 'r'))
ARCH = yaml.safe_load(open('params/arch-params.yaml', 'r'))


train = True
multi_range_gt =  True



train_dataset=SemanticKitti(root=root,sequences=DATA["split"]["train"],labels=DATA["labels"],
                            color_map=DATA["color_map"],learning_map=DATA["learning_map"],learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],multi_proj=ARCH["multi"],max_points=ARCH["dataset"]["max_points"],gt=train,multi_gt=multi_range_gt)

dataset = train_dataset[4542]



print("prog")