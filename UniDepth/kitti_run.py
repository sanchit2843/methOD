import torch

version = "v1"
backbone = "ViTL14"
model = torch.hub.load(
    "lpiccinelli-eth/UniDepth",
    "UniDepth",
    version=version,
    backbone=backbone,
    pretrained=True,
    trust_repo=True,
    force_reload=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import os

from PIL import Image
import numpy as np

os.makedirs(
    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth",
    exist_ok=True,
)
os.makedirs(
    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/points",
    exist_ok=True,
)

from tqdm import tqdm

for i in tqdm(
    os.listdir(
        "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/image_2"
    )
):
    rgb = torch.from_numpy(
        np.array(
            Image.open(
                os.path.join(
                    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/image_2",
                    i,
                )
            )
        )
    ).permute(
        2, 0, 1
    )  # C, H, W

    predictions = model.infer(rgb)

    # Metric Depth Estimation
    depth = predictions["depth"]
    points = predictions["points"]
    depth = depth.cpu().numpy()
    points = points.cpu().numpy()

    ## save point
    np.save(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth",
            i,
        ),
        depth,
    )
    np.save(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/points",
            i,
        ),
        points,
    )

    ## convert xyz points to point cloud and plot it with rgb
