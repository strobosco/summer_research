import cv2
import os
import json
import dlib
import numpy as np
from flame_model import FLAME
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Face detector model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from Dlib

# FLAME model
flame_model = FLAME("path/to/flame_model.pth")  # Load FLAME model

def extract_FLAME_parameters(video_path, learning_rate, num_iterations):
  cap = cv2.VideoCapture(video_path)
  labels = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 'frame' holds the current frame from the RGB video
    greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(greyscale_frame)

    if len(faces) == 0:
        return None  # No face detected

    landmarks = predictor(greyscale_frame, faces[0])  # Assume only one face per frame
    landmarks_2d = np.array([[p.x, p.y] for p in landmarks.parts()])  # Convert to NumPy array

    if landmarks_2d is None:
        return None  # Skip frames without detected faces

    shape_params = torch.zeros(1, 100, requires_grad=True)  # Identity
    pose_params = torch.zeros(1, 6, requires_grad=True)  # Head pose
    expression_params = torch.zeros(1, 50, requires_grad=True)  # Expressions

    # Fit FLAME to landmarks (requires optimization, simplified here)
    optimizer = optim.Adam([shape_params, pose_params, expression_params], lr=learning_rate)

    for _ in range(num_iterations):
        optimizer.zero_grad()

        # Generate 3D face from FLAME
        vertices, _ = flame_model.forward(shape_params, pose_params, expression_params)

        # Project 3D face to 2D using camera model
        projected_landmarks = project_to_2d(vertices)  # You need to implement this function

        # Compute loss (L2 distance between detected and projected landmarks)
        loss = torch.nn.functional.mse_loss(projected_landmarks, torch.tensor(landmarks_2d, dtype=torch.float32))

        # Optimize parameters
        loss.backward()
        optimizer.step()

    flame_params = torch.cat([shape_params, pose_params, expression_params], dim=1)
    labels.append(flame_params.detach().numpy())

  cap.release()
  return labels

def extract_video_frames(video_path):
  cap = cv2.VideoCapture(video_path)
  frames = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
  
    frames.append(frame)

  cap.release()
  return frames
   
def create_dataset(video_path, learning_rate, num_iterations):
  y = extract_FLAME_parameters(video_path, learning_rate, num_iterations)
  X = extract_video_frames(video_path)

  dataset = Dataset(X, y)
  return dataset

def create_dataloader(video_path, learning_rate, num_iterations, bs):
  dataset = create_dataset(video_path, learning_rate, num_iterations)
  dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
  return dataloader

# --------------------------------

# def extract_frames(video_path, output_folder, frame_rate=5):
#     os.makedirs(output_folder, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     saved_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_rate == 0:
#             frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             saved_count += 1

#         frame_count += 1

#     cap.release()
#     print(f"Extracted {saved_count} frames and saved to {output_folder}")

# # Example usage
# extract_frames("path/to/ground_truth_video.mp4", "path/to/extracted_frames", frame_rate=5)

# ###

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from Dlib

# def detect_landmarks(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) == 0:
#         return None  # No face detected

#     landmarks = predictor(gray, faces[0])  # Assume only one face per frame
#     landmarks_2d = np.array([[p.x, p.y] for p in landmarks.parts()])  # Convert to NumPy array

#     return landmarks_2d

# # Example usage
# landmarks = detect_landmarks("path/to/extracted_frames/frame_0000.jpg")
# print(landmarks)  # Shape: (68, 2)

# ###

# flame_model = FLAME("path/to/flame_model.pth")  # Load FLAME model

# def fit_flame(landmarks_2d):
#     """
#     Fits the FLAME model to 2D landmarks and returns FLAME parameters.
#     """
#     if landmarks_2d is None:
#         return None  # Skip frames without detected faces

#     shape_params = torch.zeros(1, 100)  # Initialize shape (identity) parameters
#     pose_params = torch.zeros(1, 6)  # Head pose (rotation, translation)
#     expression_params = torch.zeros(1, 50)  # Facial expressions

#     # Fit FLAME to landmarks (requires optimization, simplified here)
#     vertices, _ = flame_model.forward(shape_params, pose_params, expression_params)

#     flame_params = torch.cat([shape_params, pose_params, expression_params], dim=1)
#     return flame_params.detach().numpy()

# # Example usage
# flame_params = fit_flame(landmarks)
# print(flame_params.shape)  # Expected: (1, 156)

# ###

# output_params = {}

# frame_folder = "path/to/extracted_frames"
# flame_param_file = "path/to/flame_params.json"

# for frame in sorted(os.listdir(frame_folder)):
#     frame_path = os.path.join(frame_folder, frame)
#     landmarks = detect_landmarks(frame_path)
#     flame_params = fit_flame(landmarks)

#     if flame_params is not None:
#         output_params[frame] = flame_params.tolist()  # Convert to list for JSON storage

# # Save as JSON
# with open(flame_param_file, "w") as f:
#     json.dump(output_params, f)

# print(f"FLAME parameters saved to {flame_param_file}")