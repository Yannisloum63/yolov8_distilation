import glob
import random
import os
from shutil import copy2
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

src = '/Data/federated_learning/large_vlm_distillation_ood/datasets/coco/coco3cls/'
dst = '/Data/federated_learning/large_vlm_distillation_ood/yv8_distilation/mini_coco_dataset/'
mkdir(dst)

num_samp = 2000
tts = ['train/','val/']

for t in tts:
    path = src+'images/'+t
    src_path = src+'images/'+t
    dst_path = dst+'images/'+t
    mkdir(dst+'images/')
    mkdir(dst_path)
    mkdir(dst+'labels/')
    mkdir(dst+'labels/'+t)
    img_list = glob.glob(os.path.join(src_path,'*.jpg'))
    sampled = random.sample(img_list,num_samp)
    for item in sampled:
        copy2(item, dst_path)
        copy2(item.replace('images','labels').replace('jpg','txt'),dst_path.replace('images','labels'))
