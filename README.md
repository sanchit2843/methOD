
### Experiment Setup and Execution Guide

This repository contains code for conducting experiments on 3D object detection using the KITTI dataset. Below are detailed instructions on how to set up and run the experiment.

---

### Prerequisites

- Python 3.x installed on your system.
- GPU with CUDA support (NVIDIA A40 GPU was used in the original experiment).
- Git installed to clone the repository.
- Basic familiarity with command-line interface (CLI) and Python programming.

---

### Repository Structure

- **DAMO-YOLO**: Contains code for the 3D object detection experiment.
- **Depth2HHA-python**: Code for converting depth images to HHA representation.
- **UniDepth**: Implementation for depth estimation.
- **kitti_object_vis**: Visualization tools for KITTI dataset.
- **monodle**: MonoDepth and YOLOv3 based monocular depth estimation.
- **yolov8-kitti**: YOLOv8 model for 2D object detection on KITTI dataset.
- **check_label_format.py**: Python script for checking label format.
- **visualize_point_cloud.py**: Script to visualize point cloud data.
- **README.md**: Readme file containing instructions and information about the repository.

---

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

4. **Prepare Data**:
   - Convert depth images to HHA representation:
     ```bash
     python Depth2HHA-python/depth_to_hha.py
     ```
   - Convert labels to correct format (if necessary):
     ```bash
     python check_label_format.py
     ```

5. **Train YOLOv8 Model** (2D Object Detection):
   - Navigate to `yolov8-kitti` directory and follow instructions for training YOLOv8 model on KITTI dataset.

6. **Generate Region Proposals**:
   - Use the trained YOLOv8 model to generate region proposals.

7. **Run 3D Object Detection**:
   - Navigate to `DAMO-YOLO` directory.
   - Train the 3D object detection model:
     ```bash
     python train.py --data_path path/to/training/data --val_path path/to/validation/data --epochs 140 --lr 0.00125 --lr_decay_steps 90 120 --weight_decay 0.00001 --warmup_epochs 5 --batch_size 32
     ```

8. **Evaluate Results**:
   - Evaluate the trained model on the validation set:
     ```bash
     python evaluate.py --model_path path/to/saved/model --data_path path/to/validation/data
     ```

---

### Notes

- Experiment parameters such as learning rate, batch size, etc., can be adjusted based on hardware capabilities and experimentation requirements.
- Ensure proper directory structures and data organization to avoid errors during training and evaluation.
- Refer to individual module README files for more detailed instructions and information.

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

