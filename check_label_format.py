import os

data = open(
    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/kitti/object/training/label_2/000001.txt"
).readlines()
print("number of boxes in this image", len(data))

"""
so the format of label is that in text file we will have rows for each box 3d.
Each object will have 15 values in the row.
The values are as follows:
1. object type -- car, pedestrian, cyclist
2. truncated -- 0-1
3. occluded -- 0-3
4. alpha -- angle of the object
5. bbox -- 2d bounding box (4 values)
6. dimensions -- 3d dimensions of the object (3 values height, width, length) 
7. location -- 3d location of the object (3 values)
8. rotation_y -- rotation of the object

"""

"""
VeloScan:
Velo scans are stored in binary format. Each scan is stored as a 1D array of float32 values.
Now we need to convert this 1D array to 2D array where each row is a point in the point cloud.
The total number of columns in the 2D array will be 4. The first 3 columns will be the x, y, z coordinates of the point
The 4th column will be the intensity of the point.
"""


"""
Calibration txt:
It contains the calibration matrix of the camera. The calibration matrix is a 3x4 matrix. 4 cameras are used in the KITTI dataset.
The desired camera for us is P2. 
P2 is the camera matrix for the left color camera. It is a 3x4 matrix.
R0_rect is a 3x3 matrix which is the rectification matrix.
Tr_velo_to_cam is a 3x4 matrix which is the transformation matrix from velodyne to camera.
Tr_imu_to_velo is a 3x4 matrix which is the transformation matrix from imu to velodyne.
"""

"""
Camera and lidar axis convention:
Camera: x-right, y-down, z-forward
Lidar: x-forward, y-left, z-up


"""

import numpy as np

velo_scan = np.fromfile(
    "/home/sanchit/Workspace/courses/Gatech/Deep learning/Final_Project/kitti/object/training/velodyne/000001.bin",
    dtype=np.float32,
)

print("velo scan shape", velo_scan.shape)

scan = velo_scan.reshape((-1, 4))

print("scan shape", scan.shape)
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scan[:, 0:3])
# pcd.colors = o3d.utility.Vector3dVector(scan[:, 3:4])
## Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
