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
img_path = base_data_dir / 'image_2'
label_path = base_data_dir /'labels_2'
with open(base_data_dir / 'classes.json','r') as f:
    classes = json.load(f)
ims = sorted(list(img_path.glob('*')))
labels = sorted(list(label_path.glob('*')))
pairs = list(zip(ims,labels))
train, test = train_test_split(pairs,test_size=0.1,shuffle=True)
train_path = Path('train').resolve()
train_path.mkdir(exist_ok=True)
valid_path = Path('valid').resolve()
valid_path.mkdir(exist_ok=True)
for t_img, t_lb in tqdm(train):
    im_path = train_path / t_img.name
    lb_path = train_path / t_lb.name
    shutil.copy(t_img,im_path)
    shutil.copy(t_lb,lb_path)
for t_img, t_lb in tqdm(test):
    im_path = valid_path / t_img.name
    lb_path = valid_path / t_lb.name
    shutil.copy(t_img,im_path)
    shutil.copy(t_lb,lb_path)
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
