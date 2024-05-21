# Scene Change Detection
This project performs object detection and identifies changes between base and test images using YOLO models.

## Requirements

- `Python 3.6+`
- `numpy`
- `opencv-python`
- `ultralytics`
- `pandas`
- `argparse`

## Installation

Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

# Usage
You can run the script with the following command:

## Using Positional Arguments
```bash
python final.py base.npz test.npz
```

## Using a Mix of Positional and Optional Arguments
```bash
python final.py base.npz test.npz --custom_model path/to/custom_model.pt --pretrained_model path/to/pre_trained_mode.pt
```

## Arguments
- `base_file (positional): Path to the base .npz file.`
- `test_file (positional): Path to the test .npz file.`
- `--custom_model (optional): Path to the custom-trained YOLO model. Default is models/custom_yolo.pt.`
- `--pretrained_model (optional): Path to the pre-trained YOLO model. Default is models/yolov8n.pt.`

## Description
The script performs the following tasks:

1. Load Base and Test Images: Loads base and test images from .npz files.
2. Object Detection on Base Images: Runs object detection on base images using a custom YOLO model.
3. Object Detection on Test Images: Runs object detection on test images using both custom and pre-trained YOLO models.
4. Identify Changes: Identifies objects that have appeared, disappeared, or are unknown compared to base in test.
5. Display Results: Displays the results with bounding boxes and labels for changes.


## Example Command
```bash
python final.py base.npz test.npz --custom_model 'models/custom_yolo.pt' --pretrained_model 'models/yolov8n.pt'
```

## Custom Model Training

The custom YOLO model was trained using a dataset extracted from base.npz files. The dataset was split into 56 training images and 14 validation images. Key training parameters included 30 epochs, a batch size of 8, and an initial learning rate of 0.001 using the Adam optimizer. Various augmentation techniques were applied to enhance model robustness. The model was trained on an NVIDIA Jetson AGX Orin GPU for efficient processing.

For detailed training instructions, refer to the `/custom model training/train_yolo.py` script included in the project.

### Training Script
To train the custom YOLO model, run the following command:

```bash
cd 'custom model training'
python yolo_train.py
```
