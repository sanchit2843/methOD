## Work Done for CS7643: Deep Learning @ Georgia Institute of Technology
### Experiment Setup and Execution Guide

This repository contains code for conducting experiments on 3D object detection using the KITTI dataset. Below are detailed instructions on how to set up and run the experiment.

---

### Repository Structure

- **Depth2HHA-python**: Code for converting depth images to HHA representation.
- **UniDepth**: Implementation for depth estimation.
- **kitti_object_vis**: Visualization tools for KITTI dataset.
- **monodle**: MonoDepth and YOLOv3 based monocular depth estimation.
- yolov8-kitti: YOLOv8 model for 2D object detection on KITTI dataset.
- **README.md**: Readme file containing instructions and information about the repository.
- **monodle_multimodal** A fork of monodle folder for training multiple modalities of HHA and RGB
- **monodle_2dproposal** A fork of monodle with RPN head and support for passing proposals from dataset and to model. 
---


### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### Experiment Steps

1. **Clone Repository**:
   ```bash
   git clone <repository_url>
   ```

2. **Setup Environment**:
   - Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure CUDA and CuDNN are properly installed for GPU support.

3. **Download KITTI Dataset**:
   - Download the KITTI 3D object detection dataset from [KITTI Website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in a directory named `data` within each model repository.

4. ** Extract depth for the dataset:
   - Run inference of unidepth on RGB images for monocular depth estimation, please edit the paths in script
   ```bash
   python UniDepth/kitti_run.py
   ```
5. **Prepare Data**:
   - Convert depth images to HHA representation:
     ```bash
     python Depth2HHA-python/depth_to_hha.py
     ```

6. **Train YOLOv8 Model** (2D Object Detection):
   - Navigate to  directory and follow instructions for training YOLOv8 model on KITTI dataset.

7. **Generate Region Proposals**:
   - Use the trained YOLOv8 model to generate region proposals, they will be saved in text files.

8. **Run 3D Object Detection**:
   - For single modality model. 
   - Navigate to `mono-dle` directory, change the model type to centernet3d in experiments/example/kitti_example.yaml .
   - Change path to dataset in line 23 of lib/datasets/kitti/kitti_dataset.py 
   - In case you want to train HHA model, place those images in image_2 folder of the dataset directory other wise keep it RGB. 
   - Train the 3D object detection model:
     ```bash
     python tools/train_val.py --config experiments/example/kitti_example.yaml
     ```
   - For training multi modal, please checkout to multimodal branch, or navigate to monodle_multimodal, place rgb in image_rgb folder of the dataset folder and hha in image_hha folder, and change the model type to multi in config file.  
      ```bash
      python tools/train_val.py --config experiments/example/kitti_example.yaml
      ```
   - For evaluation update checkpoint path in kitti_example.yaml
      ```bash
      python tools/train_val.py --config experiments/example/kitti_example.yaml -e
      ```
   - For training 2d proposal based network, checkout to 2dproposal branch or navigate to monodle_2dproposal, place the prediction of boxes from yolov8 inference in "labels_without_dont_care" directory and change the model type to yolo. 
      ```
      python tools/train_val.py --config experiments/example/kitti_example.yaml
      ```

10. For point cloud visualizations, you can use the kitti_object_vis/notebook_demo.ipynb
---

---

### References

- Original paper: [Provide citation or link to the paper if available]
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [DAMO-YOLO Repository](https://github.com/tinyvision/DAMO-YOLO)
- [UniDepth Repository](https://github.com/lpiccinelli-eth/UniDepth)
- [Monodle Repository](https://github.com/xinzhuma/monodle)
- [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)

---

This README provides a comprehensive guide for setting up and running the experiment. If you encounter any issues or have questions, feel free to refer to the documentation or reach out to the repository maintainers for assistance.
