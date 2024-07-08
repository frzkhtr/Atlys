Model Selection
Annotation and Model Selection
Given the annotation type of data, the task primarily involves identifying ID cards within images for subsequent text extraction.

Suitable Models:
Faster R-CNN (Region-based Convolutional Neural Networks)
YOLO (You Only Look Once)
SSD (Single Shot MultiBox Detector)
Since I have already used YOLO and impressed by its performance, I wanted to give it another try.

Training a YOLOv3 Model for Object Detection
Training a YOLO (You Only Look Once) model for object detection on the MIDV-500 dataset involves several steps, including data preparation, model configuration, training, and evaluation. YOLO is renowned for its real-time object detection capabilities and direct prediction of bounding boxes and class probabilities using a single network evaluation. Here’s a high-level guide on how to train a YOLOv3 model on my dataset using Python and frameworks like TensorFlow or PyTorch.

Steps to Train YOLOv3 for Object Detection
Data Preparation: First, ensure my dataset is organized and annotated correctly. YOLO typically requires annotations in a specific format (like Pascal VOC or COCO format) that includes bounding box coordinates and class labels. Here’s an outline of what I need:

Images: TIFF images stored in a directory structure.
Annotations: JSON files or XML files containing bounding box coordinates for each image.
Model Configuration: I have two primary options for implementing YOLOv3:

Using Darknet Framework (C-based): The original framework for YOLO, which offers comprehensive support for training and inference.
Using a Deep Learning Framework (e.g., TensorFlow, PyTorch): Provides more flexibility and integration with other deep learning tasks.
I have used second option in model configuration.

Detailed Steps
Convert Annotations First, we will a Python script to convert the JSON annotations to YOLO format and save it in .txt format. The YOLO format requires annotations to be in the format:
<class_id> <x_center> <y_center> <width> <height>
where x_center, y_center, width, and height are normalized to the image dimensions.

import os
import json

def convert_annotation(json_file, txt_file, image_width, image_height):
    with open(json_file, 'r') as f:
        data = json.load(f)
        # print((data))

    with open(txt_file, 'w') as f:
        # print(data['quad'])
        # for annotation in data['quad']:
        annotation = data['quad']
        # Extract bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
        x1, y1 = annotation[0]
        x2, y2 = annotation[1]
        x3, y3 = annotation[2]
        x4, y4 = annotation[3]

        # Calculate bounding box center and size
        x_center = (x1 + x2 + x3 + x4) / 4.0 / image_width
        y_center = (y1 + y2 + y3 + y4) / 4.0 / image_height
        width = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) / image_width
        height = (max(y1, y2, y3, y4) - min(y1, y2, y3, y4)) / image_height

        # Write to file in YOLO format
        class_id = 0  # Assuming single class, update if you have multiple classes
        f.write(f"{class_id} {x_center} {y_center} {width} {height}
")

def process_dataset(dataset_path):
    for label_folder in os.listdir(dataset_path):
        label_folder_path = os.path.join(dataset_path, label_folder)
        if not os.path.isdir(label_folder_path):
            continue

        images_folder = os.path.join(label_folder_path, 'images')
        groundtruth_folder = os.path.join(label_folder_path, 'ground_truth')
        if not os.path.exists(images_folder) or not os.path.exists(groundtruth_folder):
            continue

        for doc_type_folder in os.listdir(images_folder):
            doc_type_folder_path = os.path.join(images_folder, doc_type_folder)
            gt_doc_type_folder_path = os.path.join(groundtruth_folder, doc_type_folder)
            if not os.path.isdir(doc_type_folder_path) or not os.path.isdir(gt_doc_type_folder_path):
                continue

            for image_file in os.listdir(doc_type_folder_path):
                if not image_file.endswith('.tif'):
                    continue

                image_path = os.path.join(doc_type_folder_path, image_file)
                json_file = os.path.join(gt_doc_type_folder_path, image_file.replace('.tif', '.json'))
                txt_file = os.path.join(gt_doc_type_folder_path, image_file.replace('.tif', '.txt'))
                txt_file = txt_file.replace('ground_truth', 'images')

                # Load image to get dimensions
                image = plt.imread(image_path)
                image_height, image_width = image.shape[:2]
                # print(json_file)
                # print(text_file)
                # Convert annotation
                convert_annotation(json_file, txt_file, image_width, image_height)

dataset_path = '.\midv500_data\midv500'
process_dataset(dataset_path)

2. Prepare Dataset for YOLO
we will copy all images and annotations in a directory:

import os
import shutil

# Define paths
current_dir = './'  # Update with your current directory path
data_dir = os.path.join(current_dir, 'data')
images_dir = os.path.join(data_dir, 'images')
midv500_dir = 'D:\playground\mdfv500\midv500_data\midv500'  # Update with your actual path to midv500

# Ensure data directory exists, create images directory
os.makedirs(images_dir, exist_ok=True)

# Function to copy TIFF images from source to destination
def copy_images(source_dir, dest_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tiff'):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                shutil.copyfile(src_file, dest_file)
                print(f'Copied {src_file} to {dest_file}')

# Copy TIFF images from midv500/images to data/images
copy_images(os.path.join(midv500_dir, '01_alb_id/images'), images_dir)
copy_images(os.path.join(midv500_dir, '02_bra_passport/images'), images_dir)
# Add more lines for other folders as needed

We'll generate train.txt and val.txt files:

import os
import shutil
import random

def create_dataset_files(dataset_path, output_path, split_ratio=0.8):
    """
    Create train.txt and val.txt files for a dataset located at dataset_path,
    with images in 'images' subfolder and labels in 'labels' subfolder,
    and save these lists to output_path.

    Parameters:
    - dataset_path (str): Path to the dataset directory.
    - output_path (str): Path to save the train.txt and val.txt files.
    - split_ratio (float): Ratio of training images to total images (default is 0.8).
    """
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    # List all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif')]
    num_images = len(image_files)
    num_train = int(num_images * split_ratio)

    # Randomize the order of images
    random.shuffle(image_files)

    # Split into training and validation sets
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # Write paths to train.txt and val.txt
    with open(os.path.join(output_path, 'train.txt'), 'w') as f:
        for file in train_files:
            image_path = os.path.join(images_dir, file)
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            f.write(f"{image_path}
")
            # Optionally check if label file exists
            if not os.path.exists(label_path):
                print(f"Warning: Label file '{label_path}' not found for '{image_path}'")

    with open(os.path.join(output_path, 'val.txt'), 'w') as f:
        for file in val_files:
            image_path = os.path.join(images_dir, file)
            label_file = os.path.splitext(file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            f.write(f"{image_path}
")
            # Optionally check if label file exists
            if not os.path.exists(label_path):
                print(f"Warning: Label file '{label_path}' not found for '{image_path}'")

# Example usage:
dataset_path = '.\data'
output_path = '.\'  # Update with your desired output path
create_dataset_files(dataset_path, output_path)

Download YOLOv5
we will go one back from current directory and clone yolov5 repo. And install its requirements in our enviornment:

cd ..
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

Prepare Configuration Files
Since we have already created your dataset files (train.txt and val.txt), the next steps are to create the configuration files for YOLOv5.

Create data.yaml This file contains the paths to your dataset and the number of classes:

train: ../atlys/data/train.txt
val: ../atlys/data/val.txt

nc: 1  # number of classes
names: ['document']  # class names

Training the Model
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name midv500_yolov5 --device 0

--img 640: Image size.
--batch 16: Batch size.
--epochs 50: Number of epochs.
--data ../app/data.yaml: Path to your data.yaml file.
--cfg models/yolov5s.yaml: Model configuration file.
--weights yolov5s.pt: Pre-trained weights file.
--name midv500_yolov5: Name of the training run.
--device: passing cuda gpu
Accuracy is very impressive able to get bounded box.
