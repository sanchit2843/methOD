from ultralytics import YOLO
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image

os.environ['WANDB_DISABLED'] = "True"
base_path = '/Users/shuttlesworthneo/Depository/OMSCS/Deep Learning/project/methOD/yolov8-kitti'
base_data_dir = Path(f'{base_path}/data/KITTI/training')
with open(base_data_dir / 'classes.json','r') as f:
    classes = json.load(f)
train_path = Path('train').resolve()
train_path.mkdir(exist_ok=True)
valid_path = Path('val').resolve()
valid_path.mkdir(exist_ok=True)
for x in ['train', 'val']:
    split_file = os.path.join(f'{base_path}/data/KITTI/', "ImageSets", f"{x}.txt")
    idx_list = [x.strip() for x in open(split_file).readlines()]
    for f_name in idx_list:
        shutil.copy(f'{base_path}/data/KITTI/training/image_2/{f_name}.png',f'{base_path}/{x}/{f_name}.png')
        shutil.copy(f'{base_path}/data/KITTI/training/label_2/{f_name}.txt', f'{base_path}/{x}/{f_name}.txt')
yaml_file = 'names:\n'
yaml_file += '\n'.join(f'- {c}' for c in classes)
yaml_file += f'\nnc: {len(classes)}'
yaml_file += f'\ntrain: {str(train_path)}\nval: {str(valid_path)}'
with open('kitti.yaml','w') as f:
    f.write(yaml_file)
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
train_results = model.train(
    data=f'{base_path}/kitti.yaml', 
    epochs=50,
    patience=3,
    mixup=0.1,
    project='yolov8n-kitti',
    device="mps"
)
valid_results = model.val()
