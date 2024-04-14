import os
import cv2
import numpy as np
from getHHA import getHHA
from utils.getCameraParam import getCameraParam

# D = cv2.imread(os.path.join(root, "0.png"), cv2.COLOR_BGR2GRAY) / 10000
# RD = cv2.imread(os.path.join(root, "0_raw.png"), cv2.COLOR_BGR2GRAY) / 10000
# camera_matrix = getCameraParam("color")
# hha = getHHA(camera_matrix, D, RD)

import matplotlib.pyplot as plt

os.makedirs(
    os.path.join(
        "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/hha"
    ),
    exist_ok=True,
)

from tqdm import tqdm

# for i in tqdm(
#     os.listdir(
#         "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth/"
#     )
# ):
#     depth = np.load(
#         os.path.join(
#             "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth/",
#             i,
#         )
#     )[0, 0]
#     calib = open(
#         os.path.join(
#             "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/calib/",
#             i.replace(".png.npy", ".txt"),
#         ),
#         "r",
#     ).readlines()

#     # ## convert kitti calib txt to 3*3 projection matrix
#     P2 = np.array(calib[2].split(": ")[1].strip().split(" ")).reshape(3, 4)
#     P2 = P2[:, :3]
#     P2 = P2.astype(np.float32)
#     # # print(len(calib[2].split(": ")[1].strip().split(" ")))
#     hha = getHHA(P2, depth, depth)
#     cv2.imwrite(
#         os.path.join(
#             "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/hha/",
#             i.replace(".png.npy", ".png"),
#         ),
#         hha,
#     )

import os
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool


def process_file(i):
    depth = np.load(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth/",
            i,
        )
    )[0, 0]
    calib = open(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/calib/",
            i.replace(".png.npy", ".txt"),
        ),
        "r",
    ).readlines()

    P2 = np.array(calib[2].split(": ")[1].strip().split(" ")).reshape(3, 4)
    P2 = P2[:, :3]
    P2 = P2.astype(np.float32)

    hha = getHHA(P2, depth, depth)
    cv2.imwrite(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/hha/",
            i.replace(".png.npy", ".png"),
        ),
        hha,
    )


if __name__ == "__main__":
    file_list = os.listdir(
        "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/depth/"
    )

    for i in os.listdir(
        "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/hha/"
    ):
        file_list.remove(i.replace(".png", ".png.npy"))

    print(len(file_list))
    with Pool(6) as pool:
        list(tqdm(pool.imap(process_file, file_list), total=len(file_list)))
