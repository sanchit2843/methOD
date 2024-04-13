import open3d as o3d
import numpy as np
import cv2
import os

for i in os.listdir(
    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/points/"
):
    points = np.load(
        os.path.join(
            "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/points/",
            i,
        )
    )

    img = cv2.imread(
        "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/KITTI/object/training/image_2/"
        + i[:-4]
    )
    ## make rgbd point cloud from rgb and points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[0].transpose(1, 2, 0).reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3) / 255.0)

    ## visualize
    o3d.visualization.draw_geometries([pcd])
