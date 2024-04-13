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

from tqdm import tqdm
import open3d as o3d
import cv2

list_ = os.listdir(
    "/home/sanchit/Workspace/clutterbot/3d_reconsutrction/dataset/zedcam/color"
)

## sort
list_ = sorted(list_, key=lambda x: int(x.split(".")[0]))[::30]

for i in tqdm(list_):
    img = cv2.imread(
        "/home/sanchit/Workspace/clutterbot/3d_reconsutrction/dataset/zedcam/color/" + i
    )
    rgb = torch.from_numpy(
        np.array(
            Image.open(
                os.path.join(
                    "/home/sanchit/Workspace/clutterbot/3d_reconsutrction/dataset/zedcam/color",
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[0].transpose(1, 2, 0).reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.0)

    ## visualize
    o3d.visualization.draw_geometries([pcd])
