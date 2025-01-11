import shutil
import os
import cv2
import numpy as np
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
import yaml
from tqdm import tqdm
from math import cos, sin, atan2, asin, sqrt, pi

# Load the 3DDFA-V2 model configuration
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Initialize FaceBoxes and TDDFA
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)

# Function to calculate yaw, pitch, and roll angles
def P2sRt(P):
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle(R):
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))
    return x, y, z

def is_front_face(yaw, pitch, roll, yaw_threshold=0.15, pitch_threshold=0.15, roll_threshold=0.5):
    return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and abs(roll) < roll_threshold

# Define main function to process images and copy them
def process_images(input_dir, output_dir):
    for root, dirs, files in tqdm(os.walk(input_dir), desc="Processing folders", unit="folder"):
        front_face_images = []
        for file in tqdm(files, desc=f"Processing files in {root}", unit="file", leave=False):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            boxes = face_boxes(image)
            if len(boxes) == 0:
                continue
            try:
                param_lst, roi_box_lst = tddfa(image, boxes)
                P = param_lst[0][:12].reshape(3, -1).copy()
                _, R, _ = P2sRt(P)
                yaw, pitch, roll = matrix2angle(R)
                
                if is_front_face(yaw, pitch, roll):
                    front_face_images.append(image_path)
                    if len(front_face_images) == 3:
                        break
            except Exception as e:
                print(f"Skipping {image_path}: {str(e)}")

        if front_face_images:
            # Preserve folder structure
            rel_path = os.path.relpath(root, input_dir)
            target_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)

            # Copy images
            for image_path in front_face_images:
                shutil.copy(image_path, target_subdir)
            #print(f"Copied {len(front_face_images)} front-facing images from {root} to {target_subdir}")

# Define input and output paths
input_dir = '/media/avlab/reggie/Paul_sd35/3DDFA_V2/prompt_generator/cvpr2025_false_pair/agedb_error_sample_all_id'
output_dir = '/media/avlab/reggie/Paul_sd35/3DDFA_V2/prompt_generator/cvpr2025_false_pair/agedb_front'

# Run the function
process_images(input_dir, output_dir)
