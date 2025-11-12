# -*- coding: utf-8 -*-
"""
Multi-camera 3D point tracking and calibration using DearPyGui (interface) and OpenCV

It loads multiple synchronised videos, allows for manual annotation of points, tracks these points using
Lucas-Kanade optical flow, and uses a genetic algorithm to solve for camera intrinsics, extrinsics, and
distortion parameters. The final output is a 3D reconstruction of the tracked points.
"""
import cv2
import numpy as np
import pathlib
import pickle
import json
import itertools
import re
from tqdm import tqdm
from typing import List, Tuple, Optional
from viz_3d import SceneObject, SceneVisualizer, create_camera_visual
import dearpygui.dearpygui as dpg
import tkinter as tk
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from segment_anything import sam_model_registry, SamPredictor
import torch
import sleap_io
import trimesh
import os  # Added for CoTracker
import torch.nn.functional as F  # Added for CoTracker
from cotracker.predictor import CoTrackerPredictor  # Added for CoTracker

from calibration import (
    CameraParams,
    flat_individual,
    unflat_individual,
    get_projection_matrix,
    undistort_points,
    combination_triangulate,
    reproject_points,
    create_individual,
    fitness,
    permutation_optimisation,
    calculate_all_reprojection_errors
)

from mesh import (
    get_sam_segmentation,
    translate_rotate_mesh_3d,
    reproject_mesh_segmentation_as_contour,
    resample_contour,
    calculate_mesh_poses_from_axis,
    scatter_pts_between,
    get_intersections_with_mesh_surface,
    calculate_change_between_poses,
    pose_vec_to_matrix,
    find_optimal_roll
)

np.set_printoptions(precision=3, suppress=True, linewidth=120)

# --- Configuration ---
DATA_FOLDER = pathlib.Path.cwd() / 'data'
VIDEO_FORMAT = '*.mp4'
SKELETON = {
        "thorax": [ "neck", "leg_f_L0", "leg_f_R0", "leg_m_L0", "leg_m_R0" ],
        "neck": [ "thorax", "a_R0", "a_L0", "eye_L", "eye_R", "m_L0", "m_R0" ],
        "eye_L": [ "neck" ],
        "eye_R": [ "neck" ],
        "a_L0": [ "neck", "a_L1" ],
        "a_L1": [ "a_L2", "a_L0" ],
        "a_L2": [ "a_L1" ],
        "a_R0": [ "neck", "a_R1" ],
        "a_R1": [ "a_R2", "a_R0" ],
        "a_R2": [ "a_R1" ],
        "leg_f_L0": [ "thorax", "leg_f_L1" ],
        "leg_f_L1": [ "leg_f_L2", "leg_f_L0" ],
        "leg_f_L2": [ "leg_f_L1" ],
        "leg_f_R0": [ "thorax", "leg_f_R1" ],
        "leg_f_R1": [ "leg_f_R2", "leg_f_R0" ],
        "leg_f_R2": [ "leg_f_R1" ],
        "leg_m_L0": [ "thorax", "leg_m_L1" ],
        "leg_m_L1": [ "leg_m_L2", "leg_m_L0" ],
        "leg_m_L2": [ "leg_m_L1" ],
        "leg_m_R0": [ "thorax", "leg_m_R1" ],
        "leg_m_R1": [ "leg_m_R2", "leg_m_R0" ],
        "leg_m_R2": [ "leg_m_R1" ],
        "m_L0": [ "neck", "m_L1" ],
        "m_L1": [ "m_L0" ],
        "m_R0": [ "neck", "m_R1" ],
        "m_R1": [ "m_R0" ],
        "s_small": [ "s_large" ],
        "s_large": []
}
GROUND_PLANE_POINTS = [
    "leg_m_R2", "leg_m_L2", "leg_f_R2", "leg_f_L2",
]
GROUND_PLANE_INDICES = np.array([list(SKELETON.keys()).index(p) for p in GROUND_PLANE_POINTS if p in SKELETON.keys()]) # (len(GROUND_PLANE_POINTS),)
POINT_NAMES = list(SKELETON.keys())
NUM_POINTS = len(POINT_NAMES)
GRID_COLS = 3  # Fixed number of columns to display the videos etc. for now

# Genetic algorithm parameters
POPULATION_SIZE = 200
ELITISM_RATE = 0.1 # Keep the top 10%

# Genetic algorithm state
mean_params = None
generation = 0
train_ga = False
best_fitness_so_far = float('inf')  # Initialize to a very high value
best_individual = None

# --- Global State ---
video_names = []
video_captures = []
video_metadata = {
    'width': 0,
    'height': 0,
    'num_frames': 0,
    'num_videos': 0,
    'fps': 30
}
current_frames = []
show_cameras = True
show_seed_only = False

# Data Structures
# Shape: (num_frames, num_camera, num_points, 2) for (x, y) coordinates
# Using np.nan for un-annotated points
annotations = None
human_annotated = None  # (num_frames, num_camera, num_points) boolean array indicating if a point is annotated
calibration_frames = []  # Frames selected for calibration, empty if not set

# SLEAP annotations and annotation visibility flags
sleap_annotations = None
show_manual_annotations = True
show_sleap_annotations = False

# Shape: (num_frames, num_points, 3) for (X, Y, Z) coordinates
reconstructed_3d_points = None
needs_3d_reconstruction = False

# Seed mesh reconstruction state
sam_predictor = None
auto_segment_mode = False
SAM_MODEL_PATH = DATA_FOLDER / "sam_vit_b_01ec64.pth"
last_segmentation_frame = -1
seed_points_2d = []
reconstructed_seed_mesh = None
seed_mesh_poses: np.ndarray = None  # (num_frames, 6) for (tvec, rvec) which is (tx, ty, tz, rx, ry, rz)
initial_seed_axis_info = {}  # Stores the axis info of the seed mesh when initially reconstructed

# CoTracker model
cotracker_model = None
COTRACKER_CHECKPOINT_PATH = DATA_FOLDER / "scaled_offline.pth"
COTRACKER_SCALE_FACTOR = 0.3

# Seed roll estimation state
mesh_roll_estimation_mode = False
mesh_roll_points_2d = []  # 2d randomly sampled points to track on seed
mesh_roll_tracks = None  # Stores results from CoTracker
mesh_rolls: np.ndarray = None
intersection_points_3d = None
predictions_3d = {}
expected_3d_points = {}

# Ground plane state
ground_plane_mode = False
ground_plane_data = None

# UI and Control State
frame_idx = 428 #300
paused = True
selected_point_idx = 0  # Default to P1
focus_selected_point = False  # Whether to focus on the selected point in the visualisation
save_output_video = False
video_save_output = None

# Lucas-Kanade Optical Flow parameters
lk_params = dict(winSize=(9, 9),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
keypoint_tracking_enabled = False
seed_pose_tracking_enabled = False

point_colors = np.array([
    [255, 0, 0],     # P1 - Red
    [0, 255, 0],     # P2 - Green
    [0, 0, 255],     # P3 - Blue
    [255, 255, 0],   # P4 - Yellow
    [0, 255, 255],   # P5 - Cyan
    [255, 0, 255],   # P6 - Magenta
    [192, 192, 192], # P7 - Silver
    [255, 128, 0],   # P8 - Orange
    [128, 0, 255],   # P9 - Purple
    [255, 128, 128], # P10 - Light Red
    [128, 128, 0],   # P11 - Olive
    [0, 128, 128],   # P12 - Teal
    [128, 0, 128],   # P13 - Maroon
    [192, 128, 128], # P14 - Salmon
    [128, 192, 128], # P15 - Light Green
    [128, 128, 192], # P16 - Light Blue
    [192, 192, 128], # P17 - Khaki
    [192, 128, 192], # P18 - Plum
    [128, 192, 192], # P19 - Light Cyan
    [255, 255, 255], # P20 - White
    [0, 0, 0],       # P21 - Black
    [128, 128, 128], # P22 - Gray
    [255, 128, 64],  # P23 - Light Orange
    [128, 64, 255],  # P24 - Light Purple
    [210, 105, 30],  # P25 - Chocolate
    [128, 255, 64],  # P26 - Light Yellow
    [128, 64, 0],    # P27 - Brown
    [64, 128, 255]   # P28 - Light Blue
], dtype=np.uint8)
assert NUM_POINTS <= len(point_colors), "Not enough colours defined for the number of points."


def load_videos():
    """Loads all videos from the specified data folder."""
    global annotations, reconstructed_3d_points, human_annotated, sleap_annotations
    global seed_points_2d, seed_mesh_poses, ground_plane_data
    video_paths = sorted(DATA_FOLDER.glob(VIDEO_FORMAT))
    if not video_paths:
        print(f"Error: No videos found in '{DATA_FOLDER}/' with format '{VIDEO_FORMAT}'")
        exit()

    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Error: Could not open video {path}")
            continue
        video_captures.append(cap)
        video_names.append(path.name)

    if not video_captures:
        print("Error: No videos were loaded successfully.")
        exit()

    # Get metadata from the first video (assuming they are synchronized and have same properties)
    video_metadata['num_videos'] = len(video_captures)
    video_metadata['width'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    video_metadata['height'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_metadata['num_frames'] = int(video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
    video_metadata['fps'] = video_captures[0].get(cv2.CAP_PROP_FPS)

    # Initialise data structures based on metadata
    annotations = np.full((video_metadata['num_frames'], video_metadata['num_videos'], NUM_POINTS, 2), np.nan, dtype=np.float32)
    reconstructed_3d_points = np.full((video_metadata['num_frames'], NUM_POINTS, 3), np.nan, dtype=np.float32)
    human_annotated = np.zeros((video_metadata['num_frames'], video_metadata['num_videos'], NUM_POINTS), dtype=bool)
    seed_points_2d = [[] for _ in range(video_metadata['num_videos'])]
    seed_mesh_poses = np.zeros((video_metadata['num_frames'], 6), dtype=np.float32)  # (num_frames, 6) for (tx, ty, tz, rx, ry, rz)
    ground_plane_data = {
        'frame': -1,
        'points_2d': [[] for _ in range(video_metadata['num_videos'])],
        'plane_model': None
    }
    sleap_annotations = np.full((video_metadata['num_frames'], video_metadata['num_videos'], NUM_POINTS, 2), np.nan, dtype=np.float32)
    print(f"Loaded {video_metadata['num_videos']} videos.")
    print(f"Resolution: {video_metadata['width']}x{video_metadata['height']}, Frames: {video_metadata['num_frames']}")

def save_state():
    """Saves the current state of annotations and 3D points."""
    np.save(DATA_FOLDER / 'annotations.npy', annotations)
    np.save(DATA_FOLDER / 'human_annotated.npy', human_annotated)
    np.save(DATA_FOLDER / 'reconstructed_3d_points.npy', reconstructed_3d_points)
    json.dump(calibration_frames, open(DATA_FOLDER / 'calibration_frames.json', 'w'))
    if best_individual is not None:
        with open(DATA_FOLDER / 'best_individual.pkl', 'wb') as f:
            pickle.dump(best_individual, f)
    if ground_plane_data is not None:
        with open(DATA_FOLDER / 'ground_plane.pkl', 'wb') as f:
            pickle.dump(ground_plane_data, f)
    np.save(DATA_FOLDER / 'seed_mesh_poses.npy', seed_mesh_poses)
    with open(DATA_FOLDER / 'initial_seed_axis_info.pkl', 'wb') as f:
        pickle.dump(initial_seed_axis_info, f)
    with open(DATA_FOLDER / 'reconstructed_seed_mesh.pkl', 'wb') as f:
        pickle.dump(reconstructed_seed_mesh, f)
    print("State saved successfully.")

def load_state():
    """Loads the saved state of annotations and 3D points."""
    global annotations, human_annotated, reconstructed_3d_points, best_individual, best_fitness_so_far, calibration_frames, ground_plane_data
    global seed_mesh_poses, initial_seed_axis_info, reconstructed_seed_mesh
    try:
        annotations = np.load(DATA_FOLDER / 'annotations.npy')
        human_annotated = np.load(DATA_FOLDER / 'human_annotated.npy')
        with open(DATA_FOLDER / 'calibration_frames.json', 'r') as f:
            calibration_frames = json.load(f)
        with open(DATA_FOLDER / 'best_individual.pkl', 'rb') as f:
            best_individual = pickle.load(f)
        best_fitness_so_far = fitness(best_individual, annotations, calibration_frames, human_annotated)  # Recalculate fitness
        reconstructed_3d_points = np.load(DATA_FOLDER / 'reconstructed_3d_points.npy')
        with open(DATA_FOLDER / 'ground_plane.pkl', 'rb') as f:
            ground_plane_data = pickle.load(f)
        seed_mesh_poses[:] = np.load(DATA_FOLDER / 'seed_mesh_poses.npy')
        with open(DATA_FOLDER / 'initial_seed_axis_info.pkl', 'rb') as f:
            initial_seed_axis_info = pickle.load(f)
        with open(DATA_FOLDER / 'reconstructed_seed_mesh.pkl', 'rb') as f:
            reconstructed_seed_mesh = pickle.load(f)
        print("State loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading state: {e}")

def initialise_sam_model():
    """Loads the SAM model into memory and prepares it for inference."""
    global sam_predictor
    if not SAM_MODEL_PATH.exists():
        print(f"SAM model checkpoint not found at '{SAM_MODEL_PATH}'.")
        print("Please download it and place it in the 'data' folder.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_MODEL_PATH))
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        sam_predictor = None

def initialise_cotracker_model():
    """Loads the CoTracker model into memory."""
    global cotracker_model
    if not COTRACKER_CHECKPOINT_PATH.exists():
        dpg.set_value("status_message", f"Error: cotracker checkpoint not found at {COTRACKER_CHECKPOINT_PATH}.")
        dpg.show_item("status_message")
        return
    try:
        cotracker_model = CoTrackerPredictor(checkpoint=COTRACKER_CHECKPOINT_PATH)
        if torch.cuda.is_available():
            cotracker_model = cotracker_model.cuda()
        print("CoTracker model loaded successfully.")
    except Exception as e:
        print(f"Error loading CoTracker model: {e}")
        cotracker_model = None
        dpg.set_value("status_message", f"Error loading CoTracker model: {e}")
        dpg.show_item("status_message")

# --- Tracking ---

def track_points(prev_gray, current_gray, cam_idx):
    """Tracks points from previous frame to current frame using Lucas-Kanade."""
    global annotations
    # Get points from the previous frame that are valid
    p0 = annotations[frame_idx - 1, cam_idx, :, :]
    valid_points_indices = ~np.isnan(p0).any(axis=1)
    if focus_selected_point:
        # Only track the selected point
        valid_points_indices[:] = False
        valid_points_indices[selected_point_idx] = True
    if not np.any(valid_points_indices):
        return
    p0_valid = p0[valid_points_indices].reshape(-1, 1, 2)
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0_valid, None, **lk_params)
    # Update annotations with tracked points
    if p1 is not None and st.any():
        good_new = p1[st == 1]
        # Get original indices of good points
        original_indices = np.where(valid_points_indices)[0]
        good_original_indices = original_indices[st.flatten() == 1]
        for i, idx in enumerate(good_original_indices):
            # Only update if the point is not already manually annotated in the current frame
            if np.isnan(annotations[frame_idx, cam_idx, idx]).any() or not human_annotated[frame_idx, cam_idx, idx]:
                annotations[frame_idx, cam_idx, idx] = good_new[i]

def run_mesh_roll_tracking(start_frame: int, points_2d: np.ndarray, cam_idx: int, num_frames: int = 50):
    """Tracks 2D points over a video clip using CoTracker."""
    global cotracker_model, mesh_roll_tracks, video_captures, video_metadata, mesh_rolls, seed_mesh_poses, intersection_points_3d, best_individual, predictions_3d
    if cotracker_model is None:
        initialise_cotracker_model()
        if cotracker_model is None:
            return
    dpg.set_value("status_message", f"Running CoTracker for {num_frames} frames.")
    dpg.show_item("status_message")
    dpg.show_item("loading_indicator")
    end_frame = min(start_frame + num_frames, video_metadata['num_frames'])
    num_frames_to_track = end_frame - start_frame
    if num_frames_to_track <= 0:
        dpg.set_value("status_message", "Error: No frames to track.")
        dpg.show_item("status_message")
        dpg.hide_item("loading_indicator")
        return

    frames_to_track = []
    cap = video_captures[cam_idx]
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(num_frames_to_track):  # Extract appropriate frames
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {start_frame + f}. Moving onto trackking.")
            break
        frames_to_track.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)  # Restore original video pos
    
    # Update num_frames_to_track in case read failed early
    num_frames_to_track = len(frames_to_track)
    if num_frames_to_track == 0:
        dpg.set_value("status_message", "Error: Could not read any frames for tracking.")
        dpg.show_item("status_message")
        dpg.hide_item("loading_indicator")
        return

    # Downsample video frames
    video_tensor = torch.from_numpy(np.stack(frames_to_track)).permute(0, 3, 1, 2)[None].float() # (1, T, C, H, W)
    B, T, C, H, W = video_tensor.shape
    video_tensor_reshaped = video_tensor.view(B * T, C, H, W)
    video_tensor_downsampled = F.interpolate(video_tensor_reshaped, scale_factor=COTRACKER_SCALE_FACTOR, mode='bilinear', align_corners=False)
    _, C_new, H_new, W_new = video_tensor_downsampled.shape
    video_tensor_downsampled = video_tensor_downsampled.view(B, T, C_new, H_new, W_new)
    print(f"Downsampled video from {video_tensor.shape} to {video_tensor_downsampled.shape}")
    
    # Create 3D scaled queries for model from 2D points    
    scaled_points_np = (np.array(points_2d) + 0.5) * COTRACKER_SCALE_FACTOR - 0.5
    queries_list = [[0, p[0], p[1]] for p in scaled_points_np]
    queries = torch.tensor(queries_list).float()[None] # (1, N, 3)
    if torch.cuda.is_available():
        video_tensor_downsampled = video_tensor_downsampled.cuda()
        queries = queries.cuda()

    try:
        with torch.no_grad():
            pred_tracks, pred_visibility = cotracker_model(video_tensor_downsampled, queries=queries, backward_tracking=True)  # (1, T, N, 2), (1, T, N)
        # Upscale tracks
        pred_tracks = pred_tracks.squeeze(0).cpu().numpy() # (T, N, 2)
        pred_visibility = pred_visibility.squeeze(0).cpu().numpy() # (T, N)
        pred_tracks_upscaled = (pred_tracks + 0.5) / COTRACKER_SCALE_FACTOR - 0.5
        mesh_roll_tracks = {
            'start_frame': start_frame,
            'cam_idx': cam_idx,
            'tracks': pred_tracks_upscaled, # (T, N, 2)
            'visibility': pred_visibility # (T, N)
        }
        print("Tracking complete.")
        dpg.set_value("status_message", f"Seed points tracking complete. Found {len(points_2d)} tracks.")
        dpg.show_item("status_message")

        start_pose = seed_mesh_poses[start_frame]  # Pose of seed relative to its pose when it was first reconstructed
        for t in range(1, num_frames_to_track):
            f = start_frame + t
            if f >= video_metadata['num_frames']:
                break
            change_pose = calculate_change_between_poses(start_pose, seed_mesh_poses[f])
            expected_3d_points[f] = translate_rotate_mesh_3d(change_pose, intersection_points_3d)
            points_2d_predictions = mesh_roll_tracks['tracks'][t]  # TODO: consider filtering these for which are visible?
            predictions_3d[f], intersection_mask = get_intersections_with_mesh_surface(best_individual[cam_idx], points_2d_predictions, reconstructed_seed_mesh, seed_mesh_poses[f])
            expected_3d_points[f] = expected_3d_points[f][intersection_mask]
            # Estimate the optimal roll angle of the seed that minimises the discrepency between the query and tracked points
            s_small_3d = reconstructed_3d_points[f, POINT_NAMES.index('s_small')]
            s_large_3d = reconstructed_3d_points[f, POINT_NAMES.index('s_large')]
            axis_centre = (s_small_3d + s_large_3d) / 2 # To rotate around
            axis_vec = s_large_3d - s_small_3d
            axis_vec_norm = axis_vec / np.linalg.norm(axis_vec)  # Roll axis
            roll_angle, min_err = find_optimal_roll(expected_3d_points[f], predictions_3d[f], axis_vec_norm, axis_centre)
            print("Frame", f, "Estimated roll:", np.round(np.degrees(roll_angle),2), "Err:", np.round(min_err,2))
            # Update seed pose with roll
            pose_vec_base = seed_mesh_poses[f]
            tvec_base, rvec_base = pose_vec_base[:3], pose_vec_base[3:]
            R_base, _ = cv2.Rodrigues(rvec_base)
            roll_rotation = Rotation.from_rotvec(roll_angle * axis_vec_norm)
            R_roll = roll_rotation.as_matrix() # 3x3
            R_new = R_roll @ R_base
            rvec_new, _ = cv2.Rodrigues(R_new)
            tvec_new = R_roll @ (tvec_base - axis_centre) + axis_centre
            seed_mesh_poses[f] = np.hstack((tvec_new.flatten(), rvec_new.flatten()))
    except Exception as e:
        print(f"Error during CoTracker inference or roll estimation: {e}")
        dpg.set_value("status_message", f"Error during tracking: {e}")
        dpg.show_item("status_message")
    finally:
        dpg.hide_item("loading_indicator")

# --- Camera calibration ---

def find_worst_frame():
    """Find and move to frame with the worst total reprojection error."""
    global frame_idx
    sorted_errors = calculate_all_reprojection_errors(video_metadata, video_names, POINT_NAMES, best_individual, annotations, reconstructed_3d_points)
    frame_errors = {}
    for e in sorted_errors:
        if e['frame'] in calibration_frames:
            continue
        frame_errors.setdefault(e['frame'], 0)
        frame_errors[e['frame']] += e['error']
    sorted_frame_errors = sorted(frame_errors.items(), key=lambda x: x[1], reverse=True) # (frame, total_error)
    frame_idx = sorted_frame_errors[0][0] if sorted_frame_errors else 0
    dpg.set_value("frame_slider", frame_idx)
    message = (f"Worst frames by total reprojection error (frame, error):\n"
               f"{np.round(sorted_frame_errors[:10],2)}"
    )
    dpg.set_value("status_message", message)
    dpg.show_item("status_message")

def find_worst_reprojection():
    """Find and move to frame with the worst reprojection error across all cameras and points."""
    global frame_idx, selected_point_idx
    sorted_errors = calculate_all_reprojection_errors(video_metadata, video_names, POINT_NAMES, best_individual, annotations, reconstructed_3d_points)
    mean_error = np.mean([e['error'] for e in sorted_errors])  # Across all cameras and points
    if len(sorted_errors) > 0:
        frame_idx = sorted_errors[0]['frame']
        selected_point_idx = POINT_NAMES.index(sorted_errors[0]['point'])
        dpg.set_value("frame_slider", frame_idx)
        dpg.set_value("point_combo", POINT_NAMES[selected_point_idx])
        message = (f"Mean reprojection error across all cameras and points for frame {frame_idx}: {mean_error:.2f}\n"
                   f"Worst keypoint in frame: '{POINT_NAMES[selected_point_idx]}'\n"
                   f"For camera: {sorted_errors[0]['camera']}"
        )
        dpg.set_value("status_message", message)
        dpg.show_item("status_message")

def run_genetic_step():
    """The main loop for the genetic algorithm to determine the best camera parameters."""
    global best_fitness_so_far, best_individual, generation, mean_params
    # Check if there are enough annotations
    num_annotations = np.sum(~np.isnan(annotations))
    if num_annotations < (2 * NUM_POINTS * 2): # Need at least 2 points in 2 views
        print("Not enough annotations to run calibration. Please annotate more points.")
        return None

    # Fitness evaluation
    std_dev = 0.001
    if best_individual is not None:
        mean_params = flat_individual(best_individual)  # Flatten the best individual parameters
    else:
        dpg.set_value("ga_status_text", "Finding optimal initial permutation... Re-fitting will start automatically.")
        dpg.set_value("ga_progress_text", "")
        dpg.render_dearpygui_frame()
        # Initialise the population with random individuals
        population = [create_individual(video_metadata) for _ in range(POPULATION_SIZE)]
        for i in tqdm(population, desc="Finding optimal initial permutation"):
            permutation_optimisation(i, annotations, calibration_frames, human_annotated)
        best_fitness_so_far = float('inf')  # Initialise to a very high value
        best_individual = None
        pop_fitness = np.array([fitness(ind, annotations) for ind in population])  # (Population,)
        mean_params = flat_individual(population[np.argmin(pop_fitness)])  # Get the best parameters from the population
        dpg.set_value("ga_status_text", "Running genetic algorithm...")

    num_params = mean_params.shape[0]  # Number of parameters in an individual
    noise = np.random.normal(0, std_dev, size=(POPULATION_SIZE, num_params))  # (Population, num_params)
    pop_params = noise + mean_params # Add noise to the best parameters for exploration (Population, num_params)
    fitness_scores = np.zeros(POPULATION_SIZE, dtype=np.float32)  # (Population,)
    temp_individual = create_individual(video_metadata)
    for i in range(POPULATION_SIZE):
        temp_individual = unflat_individual(pop_params[i], video_metadata['num_videos'])  # Unflatten the parameters
        fitness_scores[i] = fitness(temp_individual, annotations)  # Calculate fitness for the individual

    # Selection (elitism + tournament)
    sorted_population_indices = np.argsort(fitness_scores)  # Sort in ascending order
    if fitness_scores[sorted_population_indices[0]] < best_fitness_so_far:
        best_fitness_so_far = fitness_scores[sorted_population_indices[0]]
        best_individual = unflat_individual(pop_params[sorted_population_indices[0]], video_metadata['num_videos'])  # Update the best individual
    dpg.set_value("ga_progress_text", f"Generation {generation}: Best fitness (err): {best_fitness_so_far:.2f} Mean error: {np.nanmean(fitness_scores):.2f} SD: {np.nanstd(fitness_scores):.2f}")
    normalised_scores = (fitness_scores - np.nanmean(fitness_scores)) / (np.nanstd(fitness_scores) + 1e-8)  # Normalize scores
    # Update based on evolution strategy
    mean_params = mean_params + 0.1 * np.sum(noise * -normalised_scores[:, None], axis=0) / (POPULATION_SIZE * std_dev)  # Weighted sum of noise
    generation += 1

def update_3d_reconstruction(best_params: List[CameraParams]):
    """Use the best camera parameters to reconstruct all 3D points in the current frame."""
    proj_matrices = np.array([get_projection_matrix(i) for i in best_params])
    frame_annotations = annotations[frame_idx]  # (num_cams, num_points, 2)
    undistorted_annotations = np.full_like(frame_annotations, np.nan, dtype=np.float32)  # (num_cams, num_points, 2)
    for c in range(video_metadata['num_videos']):
        undistorted_annotations[c] = undistort_points(frame_annotations[c], best_params[c])  # (num_points, 2)
    points_3d = combination_triangulate(frame_annotations[None], proj_matrices)[0]  # (num_points, 3)
    reconstructed_3d_points[frame_idx] = points_3d  # Update the global 3D points for this frame

# --- Visualisation ---

def draw_ui(frame, cam_idx):
    """Draws UI elements on the frame."""
    if not auto_segment_mode and not ground_plane_mode:
        if best_individual is not None:
            reprojected = reproject_points(reconstructed_3d_points[frame_idx], best_individual[cam_idx])  # (num_points, 2)

        # Draw sleap predictions if visible
        if show_sleap_annotations:
            p_idxs = np.arange(NUM_POINTS) if not focus_selected_point else [selected_point_idx]
            for p_idx in p_idxs:
                point = sleap_annotations[frame_idx, cam_idx, p_idx]
                if not np.isnan(point).any():
                    # Draw a square for SLEAP annotations
                    top_left = tuple((point - 6).astype(int))
                    bottom_right = tuple((point + 6).astype(int))
                    cv2.rectangle(frame, top_left, bottom_right, point_colors[p_idx].tolist(), -1)
                    cv2.putText(frame, POINT_NAMES[p_idx], tuple(point.astype(int) + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 2)

        # Draw manual annotations if visible
        if show_manual_annotations:
            p_idxs = np.arange(NUM_POINTS) if not focus_selected_point else [selected_point_idx]
            for p_idx in p_idxs:
                point = annotations[frame_idx, cam_idx, p_idx]
                if not np.isnan(point).any():
                    if human_annotated[frame_idx, cam_idx, p_idx]:
                        cv2.circle(frame, tuple(point.astype(int)), 5 + 2, (255, 255, 255), -1) # White outline
                    cv2.circle(frame, tuple(point.astype(int)), 5, point_colors[p_idx].tolist(), -1)
                    cv2.putText(frame, POINT_NAMES[p_idx], tuple(point.astype(int) + np.array([5, -5])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 2)
                    if best_individual is None:
                        continue
                    point_2d_from_3d = reprojected[p_idx] # (2,)
                    if not np.isnan(point_2d_from_3d).any() and (point_2d_from_3d > 0).all():
                        # Draw a line from the reprojected point to the annotated point
                        cv2.line(frame, tuple(point.astype(int)), tuple(point_2d_from_3d.astype(int)), point_colors[p_idx].tolist(), 1)
                        distance = np.linalg.norm(point - point_2d_from_3d)
                        cv2.putText(frame, f"{distance:.2f}", tuple(point_2d_from_3d.astype(int) + np.array([5, -5])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_colors[p_idx].tolist(), 1)

    # Draw 3D reconstructed mesh reprojected back down to 2D cameras
    if reconstructed_seed_mesh is not None and best_individual is not None and frame_idx >= initial_seed_axis_info.get('frame_idx', -1):
        contour_2d = reproject_mesh_segmentation_as_contour(cam_idx, seed_mesh_poses[frame_idx], reconstructed_seed_mesh, best_individual)
        if contour_2d is not None and len(contour_2d) > 0:
            cv2.drawContours(frame, [contour_2d.astype(np.int32)], -1, (0, 255, 0), 1)

    # Draw 2D SAM segmentations of seed
    if auto_segment_mode and cam_idx < len(seed_points_2d) and frame_idx == last_segmentation_frame:
        contour = np.array(seed_points_2d[cam_idx], dtype=np.int32)
        if len(contour) > 5:
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 1)
            
    # Draw ground plane selected points
    if ground_plane_mode and ground_plane_data:
        for point in ground_plane_data['points_2d'][cam_idx]:
            cv2.circle(frame, tuple(np.array(point, dtype=int)), 5, (0, 255, 0), -1)

    # Draw current points for mesh roll estimation
    if mesh_roll_estimation_mode and cam_idx == 3:
        show = True
        if mesh_roll_tracks is not None:
            start_frame = mesh_roll_tracks['start_frame']
            if (frame_idx - start_frame) >= 1:
                show = False
        if show:
            for point in mesh_roll_points_2d:
                cv2.circle(frame, tuple(np.array(point, dtype=int)), 5, (0, 255, 0), -1)   

    # Draw cotracker points
    if mesh_roll_tracks is not None and cam_idx == mesh_roll_tracks['cam_idx']:
        start_frame = mesh_roll_tracks['start_frame']
        num_tracked_frames = mesh_roll_tracks['tracks'].shape[0] # T
        if start_frame <= frame_idx < start_frame + num_tracked_frames:
            track_frame_idx = frame_idx - start_frame
            tracks_at_frame = mesh_roll_tracks['tracks'][track_frame_idx, :, :] # (N, 2)
            visibility_at_frame = mesh_roll_tracks['visibility'][track_frame_idx, :] # (N,)
            # Cotracker recommend > 0.8 for "good" points
            visibility_threshold = 0.8 
            for i, point_xy in enumerate(tracks_at_frame):
                if visibility_at_frame[i] > visibility_threshold: # Check visibility
                    # Draw tracked points in red
                    cv2.circle(frame, tuple(np.array(point_xy, dtype=int)), 3, (0, 0, 255), -1)
    return frame

def draw_ground_plane(scene: list):
    """Adds the ground plane visualization to the 3D scene."""
    normal = ground_plane_data['plane_model']['normal']
    d = ground_plane_data['plane_model']['d']   
    # Create a large rectangle on the estimated plane
    if np.allclose(normal, [0, 1, 0]) or np.allclose(normal, [0, -1, 0]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    # Center of the plane (can be approximated from the points used for estimation)
    # For now, let's assume it's around the origin for simplicity
    center = -d * normal # A point on the plane
    size = 1.5 # Size of the rectangle
    p1 = center + size * u + size * v
    p2 = center - size * u + size * v
    p3 = center - size * u - size * v
    p4 = center + size * u - size * v
    color = (128, 128, 128, 200)
    scene.append(SceneObject(type='line', coords=np.array([p1, p2]), color=color, label=None))
    scene.append(SceneObject(type='line', coords=np.array([p2, p3]), color=color, label=None))
    scene.append(SceneObject(type='line', coords=np.array([p3, p4]), color=color, label=None))
    scene.append(SceneObject(type='line', coords=np.array([p4, p1]), color=color, label=None))

# --- DPG UI ---

def get_screen_dimensions():
    """Gets the screen dimensions using tkinter."""
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height

def resize_callback(sender, app_data, user_data):
    """Callback for when the viewport is resized."""
    viewport_width = app_data[0]
    video_grid_width = viewport_width - 320  # Leave space for the control panel
    aspect_ratio = video_metadata['width'] / video_metadata['height']
    item_width = (video_grid_width / GRID_COLS) - 20 # Subtract some padding
    item_height = item_width / aspect_ratio
    for i in range(video_metadata['num_videos']):
        dpg.configure_item(f"video_image_{i}", width=item_width, height=item_height)
    dpg.configure_item("3d_image", width=item_width, height=item_height)

def on_key_press(sender, app_data):
    """Key press handler."""
    allowed_keys = {dpg.mvKey_Q, dpg.mvKey_S, dpg.mvKey_P} # In autosegment mode
    if (auto_segment_mode or ground_plane_mode) and app_data not in allowed_keys:
        return
    match app_data:
        case dpg.mvKey_Spacebar:
            toggle_pause(sender, app_data, None)
        case dpg.mvKey_Left:
            prev_frame(sender, app_data, None)
        case dpg.mvKey_Right:
            next_frame(sender, app_data, None)
        case dpg.mvKey_Up:
            set_selected_point(None, POINT_NAMES[(selected_point_idx - 1) % NUM_POINTS], None)
        case dpg.mvKey_Down:
            set_selected_point(None, POINT_NAMES[(selected_point_idx + 1) % NUM_POINTS], None)
        case dpg.mvKey_T:
            toggle_keypoint_tracking()
        case dpg.mvKey_P:
            toggle_seed_pose_tracking()
        case dpg.mvKey_G:
            toggle_ga(sender, app_data, None)
        case dpg.mvKey_H:
            set_human_annotated(sender, app_data, None)
        case dpg.mvKey_D:
            clear_future_annotations(sender, app_data, None)
        case dpg.mvKey_S:
            save_state()
        case dpg.mvKey_L:
            load_state()
        case dpg.mvKey_F:
            find_worst_reprojection()
        case dpg.mvKey_W:
            find_worst_frame()
        case dpg.mvKey_C:
            add_to_calib_frames(sender, app_data, None)
        case dpg.mvKey_Q:
            dpg.stop_dearpygui()
        case dpg.mvKey_R:
            toggle_record(sender, app_data, None)
        case dpg.mvKey_Z:
            global focus_selected_point
            focus_selected_point = not focus_selected_point

def create_ga_popup():
    """Creates the popup window for the genetic algorithm."""
    with dpg.window(label="Calibration", modal=False, show=False, tag="ga_popup", width=540, height=120, on_close=toggle_ga_pause) as ga_window:
        dpg.add_text("Running genetic algorithm...", tag="ga_status_text")
        dpg.add_text("", tag="ga_progress_text")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset", callback=reset_ga)
            dpg.add_button(label="Pause", callback=toggle_ga_pause, tag="ga_pause_button")
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        window_width = dpg.get_item_width(ga_window)
        window_height = dpg.get_item_height(ga_window)
        dpg.set_item_pos(ga_window, [int((viewport_width - window_width) * 0.5), int((viewport_height - window_height) * 0.5)])

def create_dpg_ui(textures: np.ndarray, scene_viz: SceneVisualizer):
    """Creates the DearPyGui UI."""
    dpg.create_context()
    screen_width, screen_height = get_screen_dimensions()
    viewport_width = int(screen_width * 0.9)
    viewport_height = int(screen_height * 0.9)

    # Calculate the position to center the viewport
    x_pos = int((screen_width - viewport_width) * 0.5)
    y_pos = int((screen_height - viewport_height) * 0.5)

    # File dialog for SLEAP import
    with dpg.file_dialog(directory_selector=False, show=False, callback=sleap_file_callback, tag="file_dialog_id", width=700, height=400):
        dpg.add_file_extension(".slp", color=(0, 255, 0, 255))
        dpg.add_file_extension(".*")

    dpg.create_viewport(title="CATAR", width=viewport_width, height=viewport_height, x_pos=x_pos, y_pos=y_pos)
    with dpg.viewport_menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save state", callback=save_state)
            dpg.add_menu_item(label="Load state", callback=load_state)
            dpg.add_menu_item(label="Import SLEAP predictions", callback=import_sleap_predictions)
        with dpg.menu(label="Calibration"):
            dpg.add_menu_item(label="Run genetic algorithm", callback=toggle_ga)
            dpg.add_menu_item(label="Find worst calibration", callback=find_worst_frame)
            dpg.add_menu_item(label="Find worst reprojection", callback=find_worst_reprojection)
            dpg.add_menu_item(label="Add calibration frame", callback=add_to_calib_frames)
        with dpg.menu(label="Seed"):
            dpg.add_menu_item(label="Enable 2D auto-segmentation", callback=lambda: toggle_auto_segment_mode(), check=True, tag="auto_segment_checkbox")
            dpg.add_menu_item(label="Reconstruct 3D object mesh", callback=reconstruct_seed_mesh)
            dpg.add_menu_item(label="Clear segmentation points", callback=clear_seed_points)
            dpg.add_menu_item(label="Lock 3D obj mesh to 2D obj annotations", callback=lock_mesh_to_annotations)
            # dpg.add_menu_item(label="Estimate obj pose in current frame", callback=estimate_mesh_pose)
            dpg.add_menu_item(label="Estimate obj roll", callback=toggle_mesh_roll_estimation)
        with dpg.menu(label="Ground plane"):
            dpg.add_menu_item(label="Estimate ground plane", callback=toggle_ground_plane)

    create_ga_popup()
    dpg.setup_dearpygui()
    dpg.set_viewport_resize_callback(resize_callback)

    # Create textures for each video feeds and one for the 3D reprojection
    with dpg.texture_registry():
        for i in range(video_metadata['num_videos']):
            dpg.add_raw_texture(
                width=video_metadata['width'],
                height=video_metadata['height'],
                default_value=textures[i].ravel(),
                tag=f"video_texture_{i}",
                format=dpg.mvFormat_Float_rgba
            )
        dpg.add_raw_texture(
            width=video_metadata['width'],
            height=video_metadata['height'],
            default_value=textures[-1].ravel(),
            tag="3d_texture",
            format=dpg.mvFormat_Float_rgba
        )

    with dpg.theme(tag="record_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 0, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 50, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 100, 100, 255))
    with dpg.theme(tag="tracking_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 255, 0, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 255, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 255, 100, 255))

    with dpg.window(label="Main Window", tag="main_window"):
        with dpg.group(horizontal=True):
            # Left side control panel
            with dpg.child_window(width=300):
                create_control_panel()
            # Right side video and 3D projection
            with dpg.child_window(width=-1):
                with dpg.child_window(height=-150, tag="video_grid_window"):
                    create_video_grid(scene_viz)
                # Annotation histogram
                with dpg.child_window(height=150, tag="histogram_window"):
                    with dpg.plot(label="Annotation histogram", height=-1, width=-1, no_menus=True, no_box_select=True, no_mouse_pos=True, tag="annotation_plot"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="histogram_x_axis")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Annotations", tag="histogram_y_axis")
                        dpg.add_bar_series(list(range(video_metadata['num_frames'])), [0]*video_metadata['num_frames'], label="Human", parent="histogram_y_axis", tag="annotation_histogram_series")
                        dpg.add_bar_series(list(range(video_metadata['num_frames'])), [0]*video_metadata['num_frames'], label="SLEAP", parent="histogram_y_axis", tag="sleap_histogram_series")
                        dpg.add_drag_line(label="Current frame", color=[255, 0, 0, 255], vertical=True, default_value=frame_idx, tag="current_frame_line")
                    with dpg.item_handler_registry(tag="histogram_handler"):
                        dpg.add_item_clicked_handler(callback=on_histogram_click)
                    dpg.bind_item_handler_registry("annotation_plot", "histogram_handler")

    with dpg.handler_registry():
        dpg.add_key_press_handler(callback=on_key_press)
        dpg.add_mouse_wheel_handler(callback=scene_viz.dpg_on_mouse_wheel, user_data="3d_image")
        dpg.add_mouse_move_handler(callback=scene_viz.dpg_drag_move)
        dpg.add_mouse_release_handler(callback=scene_viz.dpg_drag_end)

    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()

def create_control_panel():
    """Creates the control panel on the left side of the UI."""
    dpg.add_text("--- Info ---")
    dpg.add_text(f"Frame: {frame_idx}/{video_metadata['num_frames']}", tag="frame_text")
    dpg.add_text(f"Status: {'Paused' if paused else 'Playing'}", tag="status_text")
    dpg.add_text(f"Save output video: {'Enabled' if save_output_video else 'Disabled'}", tag="save_video_text")
    dpg.add_text(f"Autosegment mode: {'Enabled' if auto_segment_mode else 'Disabled'}", tag="autosegment_text")
    dpg.add_text(f"Tracking: {'Enabled' if (keypoint_tracking_enabled or seed_pose_tracking_enabled) else 'Disabled'}", tag="tracking_text")
    dpg.add_text(f"Focus mode: {'Enabled' if focus_selected_point else 'Disabled'}", tag="focus_text")
    dpg.add_text(f"Annotating keypoint: {POINT_NAMES[selected_point_idx]}", tag="annotating_point_text")
    dpg.add_spacing(count=5)
    dpg.add_text(f"Best fitness: {best_fitness_so_far:.2f}", tag="fitness_text")
    dpg.add_text(f"Num annotation: {np.sum(~np.isnan(annotations[frame_idx])) // 2} / {NUM_POINTS * len(video_names)}", tag="num_annotations_text")
    dpg.add_text(f"Num 3D points: {np.sum(~np.isnan(reconstructed_3d_points[frame_idx]).any(axis=1))} / {NUM_POINTS}", tag="num_3d_points_text")
    dpg.add_text(f"Num calibration frames: {len(calibration_frames)}", tag="num_calib_frames_text")
    dpg.add_separator()
    dpg.add_text("--- Controls ---")
    with dpg.group(horizontal=True):
        dpg.add_button(label="< Prev", callback=prev_frame)
        play_label = "Play" if paused else "Pause"
        dpg.add_button(label=play_label, callback=toggle_pause, tag="play_pause_button")
        dpg.add_button(label="Next >", callback=next_frame)
    dpg.add_slider_int(label="Frame", min_value=0, max_value=video_metadata['num_frames'] - 1, default_value=frame_idx, callback=set_frame, tag="frame_slider")
    dpg.add_combo(label="Keypoint", items=POINT_NAMES, default_value=POINT_NAMES[selected_point_idx], callback=set_selected_point, tag="point_combo")
    dpg.add_button(label="Set all previous to 'human annotated'", callback=set_human_annotated)
    dpg.add_button(label="Delete future annotations", callback=clear_future_annotations)
    dpg.add_button(label="Track keypoints", callback=toggle_keypoint_tracking, tag="keypoint_tracking_button")
    # dpg.add_button(label="Track seed pose", callback=toggle_seed_pose_tracking, tag="seed_pose_tracking_button")
    dpg.add_button(label="Record", callback=toggle_record, tag="record_button")
    dpg.add_checkbox(label="Show histogram", default_value=True, callback=toggle_histogram)
    dpg.add_checkbox(label="Show manual annotations", default_value=True, callback=toggle_manual_annotations)
    dpg.add_checkbox(label="Show sleap annotations", default_value=False, callback=toggle_sleap_annotations)

    dpg.add_separator()
    dpg.add_text("--- Messages ---")
    dpg.add_text("", tag="status_message", color=(255, 100, 100), wrap=280, show=False)
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=125)
        dpg.add_loading_indicator(tag="loading_indicator", show=False, style=1, speed=0.5)

def create_video_grid(scene_viz: SceneVisualizer):
    """Creates the grid for video feeds and 3D projection."""
    num_items = video_metadata['num_videos'] + 1
    with dpg.table(header_row=False):
        for _ in range(GRID_COLS):
            dpg.add_table_column()
        num_rows = (num_items + GRID_COLS - 1) // GRID_COLS
        for i in range(num_rows):
            with dpg.table_row():
                for j in range(GRID_COLS):
                    idx = i * GRID_COLS + j
                    if idx < video_metadata['num_videos']:
                        with dpg.table_cell():
                            dpg.add_text(video_names[idx])
                            dpg.add_image(f"video_texture_{idx}", tag=f"video_image_{idx}")
                            with dpg.item_handler_registry(tag=f"image_handler_{idx}"):
                                dpg.add_item_clicked_handler(callback=image_click_callback, user_data=idx)
                            dpg.bind_item_handler_registry(f"video_image_{idx}", f"image_handler_{idx}")
                    elif idx == video_metadata['num_videos']:
                        with dpg.table_cell():
                            with dpg.group(horizontal=True):
                                dpg.add_text("3D Projection")
                                dpg.add_button(label="Reset view", small=True, callback=scene_viz.reset_view)
                                dpg.add_button(label="Hide cameras", small=True, callback=toggle_cameras)
                                dpg.add_button(label="Seed only", small=True, callback=toggle_show_seed_only)
                            dpg.add_image("3d_texture", tag="3d_image")
                            # Bind scene visualizer mouse events to the 3D image
                            with dpg.item_handler_registry(tag="3d_image_handler"):
                                dpg.add_item_clicked_handler(callback=scene_viz.dpg_drag_start)
                            dpg.bind_item_handler_registry("3d_image", "3d_image_handler")

# --- DPG callbacks ---

def toggle_record(sender, app_data, user_data):
    global save_output_video, video_save_output
    save_output_video = not save_output_video
    if save_output_video:
        dpg.bind_item_theme("record_button", "record_button_theme")
    else:
        dpg.bind_item_theme("record_button", 0) # 0 resets to the default theme
    if save_output_video:  # Initialise video writer
        num_rows = video_metadata["num_videos"] // GRID_COLS + (1 if video_metadata["num_videos"] % GRID_COLS > 0 else 0)
        fourcc = cv2.VideoWriter_fourcc(*'hvc1')
        video_save_output = cv2.VideoWriter("recording.mp4", fourcc, 30.0,
                                            (video_metadata['width'] * GRID_COLS,
                                             video_metadata['height'] * num_rows))

def toggle_pause(sender, app_data, user_data):
    global paused
    paused = not paused
    if paused:
        dpg.configure_item("play_pause_button", label="Play")
    else:
        dpg.configure_item("play_pause_button", label="Pause")

def next_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if frame_idx < video_metadata['num_frames'] - 1:
        frame_idx += 1
    dpg.set_value("frame_slider", frame_idx)

def prev_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if frame_idx > 0:
        frame_idx -= 1
    dpg.set_value("frame_slider", frame_idx)

def set_frame(sender, app_data, user_data):
    global frame_idx, paused
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    frame_idx = app_data

def set_selected_point(sender, app_data, user_data):
    global selected_point_idx
    selected_point_idx = POINT_NAMES.index(app_data)
    dpg.set_value("point_combo", POINT_NAMES[selected_point_idx])

def set_human_annotated(sender, app_data, user_data):
    if focus_selected_point:
        human_annotated[:frame_idx + 1, :, selected_point_idx] = True
        print(f"Marked all previous frames as human-annotated for point {POINT_NAMES[selected_point_idx]}.")

def clear_future_annotations(sender, app_data, user_data):
    if focus_selected_point:
        annotations[frame_idx:, :, selected_point_idx] = np.nan
        human_annotated[frame_idx:, :, selected_point_idx] = False
        print(f"Cleared all annotations for {POINT_NAMES[selected_point_idx]} from frame {frame_idx}")

def toggle_keypoint_tracking():
    global keypoint_tracking_enabled, seed_pose_tracking_enabled
    keypoint_tracking_enabled = not keypoint_tracking_enabled
    if keypoint_tracking_enabled:
        dpg.bind_item_theme("keypoint_tracking_button", "tracking_button_theme")
        seed_pose_tracking_enabled = False
        dpg.bind_item_theme("seed_pose_tracking_button", 0)
    else:
        dpg.bind_item_theme("keypoint_tracking_button", 0)

def toggle_seed_pose_tracking():
    global seed_pose_tracking_enabled, keypoint_tracking_enabled
    seed_pose_tracking_enabled = not seed_pose_tracking_enabled
    if seed_pose_tracking_enabled:
        dpg.bind_item_theme("seed_pose_tracking_button", "tracking_button_theme")
        keypoint_tracking_enabled = False
        dpg.bind_item_theme("keypoint_tracking_button", 0)
    else:
        dpg.bind_item_theme("seed_pose_tracking_button", 0)

def toggle_ga(sender, app_data, user_data):
    global train_ga, best_fitness_so_far, paused
    train_ga = not train_ga
    paused = True
    dpg.configure_item("play_pause_button", label="Play")
    if train_ga:
        dpg.configure_item("ga_popup", show=True)
    else:
        dpg.configure_item("ga_popup", show=False)

def toggle_ga_pause():
    global train_ga
    train_ga = not train_ga

def reset_ga():
    global generation, best_fitness_so_far, best_individual
    generation = 0
    best_fitness_so_far = float('inf')
    best_individual = None

def toggle_cameras():
    """Toggles the visibility of the cameras in the 3D view."""
    global show_cameras, needs_3d_reconstruction
    show_cameras = not show_cameras
    needs_3d_reconstruction = True  # Trigger a redraw of the scene

def toggle_show_seed_only():
    """Toggles the visibility of the cameras in the 3D view."""
    global show_seed_only, needs_3d_reconstruction
    show_seed_only = not show_seed_only
    needs_3d_reconstruction = True  # Trigger a redraw of the scene

def add_to_calib_frames(sender, app_data, user_data):
    global calibration_frames
    if frame_idx not in calibration_frames:
        calibration_frames.append(frame_idx)
        print(f"Frame {frame_idx} added to calibration frames.")

def image_click_callback(sender, app_data, user_data):
    """Callback function for handling mouse clicks on video images."""
    global needs_3d_reconstruction, current_frames, sleap_annotations, annotations
    cam_idx = user_data
    image_tag = f"video_image_{cam_idx}"
    # Mouse pos in absolute window coords
    mouse_pos_abs = dpg.get_mouse_pos(local=False)
    # Get the absolute position of the top-left corner of the frame
    image_pos_abs = dpg.get_item_rect_min(image_tag)
    local_pos_x = mouse_pos_abs[0] - image_pos_abs[0]
    local_pos_y = mouse_pos_abs[1] - image_pos_abs[1]
    # Current size of frame
    image_size = dpg.get_item_rect_size(image_tag)
    # Scale local mouse position to original video's resolution
    scaled_x = local_pos_x * video_metadata['width'] / image_size[0]
    scaled_y = local_pos_y * video_metadata['height'] / image_size[1]
    # Clip coordinates within frame bounds
    mouse_pos = (max(0, min(video_metadata['width'] - 1, scaled_x)),
                 max(0, min(video_metadata['height'] - 1, scaled_y)))

    if auto_segment_mode:
        if app_data[0] == 0: # Left-click to auto-segment
            dpg.show_item("loading_indicator")
            try:
                print(f"Running auto-segmentation for camera {cam_idx} with prompt at {mouse_pos}...")
                frame_to_segment = current_frames[cam_idx] # Get current frame for segmentation
                contour_points = get_sam_segmentation(frame_to_segment, mouse_pos, sam_predictor)
                if contour_points is not None:
                    seed_points_2d[cam_idx] = contour_points.tolist() # Replace the old points with new contour
                    # print(f"Successfully segmented seed with {len(contour_points)} points.")
                    global last_segmentation_frame
                    last_segmentation_frame = frame_idx
                else:
                    print("Auto-segmentation failed.")
            finally:
                dpg.hide_item("loading_indicator")
        return # Don't allow other functionality whilst in autosegment mode

    if ground_plane_mode:
        if app_data[0] == 0 and len(ground_plane_data['points_2d'][cam_idx]) < 4:
            ground_plane_data['points_2d'][cam_idx].append(mouse_pos)
            print(f"Added ground plane point {len(ground_plane_data['points_2d'][cam_idx])}/4 for camera {cam_idx}")
            if len(ground_plane_data['points_2d'][cam_idx]) == 4:
                # Once 4 points are selected, check if we have enough data to calculate the plane
                cams_with_4_points = [i for i, pts in enumerate(ground_plane_data['points_2d']) if len(pts) == 4]
                if len(cams_with_4_points) >= 2:
                    cam1_idx, cam2_idx = cams_with_4_points[0], cams_with_4_points[1]
                    # Undistort the four 2d points for both cameras and get 3D projection coords
                    p1_2d = np.array(ground_plane_data['points_2d'][cam1_idx], dtype=np.float32)
                    p2_2d = np.array(ground_plane_data['points_2d'][cam2_idx], dtype=np.float32)
                    p1_undistorted = undistort_points(p1_2d, best_individual[cam1_idx])
                    p2_undistorted = undistort_points(p2_2d, best_individual[cam2_idx])
                    proj_matrices = np.array([get_projection_matrix(cam) for cam in best_individual])
                    points_4d_hom = cv2.triangulatePoints(proj_matrices[cam1_idx], proj_matrices[cam2_idx], p1_undistorted.T, p2_undistorted.T)
                    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
                    # Fit a plane to the 3D points
                    centroid = np.mean(points_3d, axis=0)
                    centered_points = points_3d - centroid
                    _, _, vh = np.linalg.svd(centered_points)
                    normal = vh[-1]
                    d = -np.dot(normal, centroid)
                    # Store data
                    ground_plane_data['plane_model'] = {'normal': normal, 'd': d, 'points_3d': points_3d}
                    ground_plane_data['frame'] = frame_idx
                    toggle_ground_plane() # Toggle back to normal mode
                    needs_3d_reconstruction = True
        return  # Don't allow other functionality whilst estimating ground plane

    if app_data[0] == 0:  # Left click to annotate/move
        if show_manual_annotations:
            annotations[frame_idx, cam_idx, selected_point_idx] = (float(mouse_pos[0]), float(mouse_pos[1]))
            human_annotated[frame_idx, cam_idx, selected_point_idx] = True
            print(f"Annotated {POINT_NAMES[selected_point_idx]} at ({mouse_pos[0]:.2f}, {mouse_pos[1]:.2f}) in cam {cam_idx} at frame {frame_idx}")
            needs_3d_reconstruction = True
    elif app_data[0] == 1: # Right click to remove annotation
        if show_manual_annotations:
            annotations[frame_idx, cam_idx, selected_point_idx] = np.nan
            human_annotated[frame_idx, cam_idx, selected_point_idx] = False
            print(f"Removed annotation for {POINT_NAMES[selected_point_idx]} in cam {cam_idx} at frame {frame_idx}")
            needs_3d_reconstruction = True

def toggle_auto_segment_mode():
    """Toggles the automatic seed segmentation mode."""
    global auto_segment_mode
    auto_segment_mode = not auto_segment_mode
    if auto_segment_mode and sam_predictor is None:
        initialise_sam_model() # Load the model when mode is first enabled
        if sam_predictor is None:
            auto_segment_mode = False
            dpg.set_value("status_message", "Loading of SAM2 model failed. Please check its .pth file is in 'data'.")
    dpg.set_value("auto_segment_checkbox", auto_segment_mode)
    if auto_segment_mode:
        dpg.set_value("status_message", "Left-click in a video frame to segment an object.")
    else:
        dpg.set_value("status_message", "")
    dpg.show_item("status_message")
    print(f"Auto-segment mode: {'On' if auto_segment_mode else 'Off'}")

def clear_seed_points():
    """Clears all selected sam-segmented seed points."""
    global seed_points_2d, reconstructed_seed_mesh, needs_3d_reconstruction
    for i in range(len(seed_points_2d)):
        seed_points_2d[i].clear()
    # reconstructed_seed_mesh = None
    needs_3d_reconstruction = True

def reconstruct_seed_mesh():
    """Reconstructs a 3D mesh from the 2D seed contours using triangulation
    and find the best correspondence between contour points."""
    global reconstructed_seed_mesh, needs_3d_reconstruction, initial_seed_axis_info
    if best_individual is None:
        dpg.set_value("status_message", "Error: Provide calibration before reconstructing mesh.")
        dpg.show_item("status_message")
        return

    cams_with_points = [i for i, pts in enumerate(seed_points_2d) if len(pts) > 2]
    if len(cams_with_points) < 2:
        dpg.set_value("status_message", "Error: Segment object in at least two views.")
        dpg.show_item("status_message")
        return

    proj_matrices = np.array([get_projection_matrix(cam) for cam in best_individual])
    camera_pairs = list(itertools.combinations(cams_with_points, 2))
    all_3d_points = []
    num_resampled_points = 50
    dpg.show_item("loading_indicator")
    for cam_idx_1, cam_idx_2 in camera_pairs:
        print(f"Processing pair: Camera {cam_idx_1} and Camera {cam_idx_2}")
        contour1_orig = np.array(seed_points_2d[cam_idx_1], dtype=np.float32)
        contour2_orig = np.array(seed_points_2d[cam_idx_2], dtype=np.float32)
        contour1 = resample_contour(contour1_orig, num_resampled_points)
        contour2 = resample_contour(contour2_orig, num_resampled_points)

        # Find the best rotational alignment by minimising reprojection error
        min_error = float('inf')
        best_offset = 0
        for offset in range(num_resampled_points):
            contour2_rolled = np.roll(contour2, offset, axis=0)
            contour1_undistorted = undistort_points(contour1, best_individual[cam_idx_1])
            contour2_rolled_undistorted = undistort_points(contour2_rolled, best_individual[cam_idx_2])
            points_4d_hom = cv2.triangulatePoints(proj_matrices[cam_idx_1], proj_matrices[cam_idx_2], contour1_undistorted.T, contour2_rolled_undistorted.T)
            points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
            if np.any(np.isinf(points_3d)) or np.any(np.isnan(points_3d)):
                continue # Skip if any points are at infinity

            reproj1 = reproject_points(points_3d, best_individual[cam_idx_1])
            reproj2 = reproject_points(points_3d, best_individual[cam_idx_2])
            error1 = np.linalg.norm(contour1 - reproj1, axis=1).sum()
            error2 = np.linalg.norm(contour2_rolled - reproj2, axis=1).sum()
            total_error = error1 + error2
            if total_error < min_error:
                min_error = total_error
                best_offset = offset
        print(f"  - Found best alignment with offset {best_offset} and error {min_error:.2f}")

        # Triangulate with the best alignment using undistorted points for final accuracy
        contour2_aligned = np.roll(contour2, best_offset, axis=0)
        contour1_final_undistorted = undistort_points(contour1, best_individual[cam_idx_1])
        contour2_aligned_final_undistorted = undistort_points(contour2_aligned, best_individual[cam_idx_2])

        points_4d_hom = cv2.triangulatePoints(proj_matrices[cam_idx_1], proj_matrices[cam_idx_2], contour1_final_undistorted.T, contour2_aligned_final_undistorted.T)
        points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
        all_3d_points.append(points_3d)
    if not all_3d_points:
        dpg.set_value("status_message", "Error: Triangulation failed.")
        dpg.show_item("status_message")
        return

    final_points_3d = np.vstack(all_3d_points)
    try:
        # Reconstruct seed mesh
        hull = ConvexHull(final_points_3d)
        faces = hull.simplices
        reconstructed_seed_mesh = {"points": final_points_3d, "faces": faces, "vertices": hull.vertices}
        initial_seed_axis_info = {
            'frame_idx': frame_idx,
            's_small_3d': reconstructed_3d_points[frame_idx, POINT_NAMES.index('s_small')],
            's_large_3d': reconstructed_3d_points[frame_idx, POINT_NAMES.index('s_large')]
        }
        needs_3d_reconstruction = True # Trigger redraw
        print(f"Successfully reconstructed seed into a mesh with {len(final_points_3d)} vertices and {len(faces)} faces.")
        dpg.set_value("status_message", "Seed mesh reconstructed.")
        dpg.show_item("status_message")
    except Exception as e:
        print(f"Error creating Convex Hull: {e}. Storing as point cloud.")
        reconstructed_seed_mesh = {"points": final_points_3d, "faces": []} # Store as point cloud if hull fails
        needs_3d_reconstruction = True
        dpg.set_value("status_message", "Could not form mesh, showing point cloud.")
        dpg.show_item("status_message")
    dpg.hide_item("loading_indicator")

def lock_mesh_to_annotations():
    """Locks the 3D seed mesh to the annotated seed axis throughout the video."""
    global needs_3d_reconstruction, seed_mesh_poses
    if reconstructed_seed_mesh is None:
        dpg.set_value("status_message", "Error: Reconstruct a mesh before locking it to the axis labels of the object.")
        dpg.show_item("status_message")
        return
    if 's_small' not in POINT_NAMES or 's_large' not in POINT_NAMES:
        dpg.set_value("status_message", "Error: 's_small' and 's_large' must be defined in the SKELETON.")
        dpg.show_item("status_message")
        return
    seed_mesh_poses = calculate_mesh_poses_from_axis(POINT_NAMES, ('s_small', 's_large'), initial_seed_axis_info,  video_metadata['num_frames'], seed_mesh_poses, reconstructed_3d_points)
    needs_3d_reconstruction = True

def import_sleap_predictions():
    """Handles the import of SLEAP predictions from a .slp file."""
    dpg.show_item("file_dialog_id")

def sleap_file_callback(sender, app_data):
    """Callback for the SLEAP file dialog."""
    file_path = pathlib.Path(app_data['file_path_name'])
    session_match = re.search(r'(session\d+)', file_path.name)
    if not session_match:
        dpg.set_value("status_message", "Error: No 'sessionN' string found in the filename.")
        dpg.show_item("status_message")
        return
    session_str = session_match.group(1)
    session_files = [f for f in file_path.parent.glob(f"*{session_str}*.slp")]
    if not session_files:
        dpg.set_value("status_message", f"Error: No other slp files found for {session_str}.")
        dpg.show_item("status_message")
        return
        
    all_labels = [sleap_io.load_slp(str(f)) for f in session_files]
    all_keypoints = all_labels[0].skeleton.node_names
    with dpg.window(label="Import keypoints", modal=True, show=True, tag="sleap_keypoint_selector", width=400, height=350) as sleap_window:
        dpg.add_text("Select the keypoints to import:")
        def select_all(sender, app_data, user_data):
            for keypoint in all_keypoints:
                dpg.set_value(f"kp_checkbox_{keypoint}", True)
        def deselect_all(sender, app_data, user_data):
            for keypoint in all_keypoints:
                dpg.set_value(f"kp_checkbox_{keypoint}", False)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select all", callback=select_all)
            dpg.add_button(label="Deselect all", callback=deselect_all)
        for keypoint in all_keypoints:
            dpg.add_checkbox(label=keypoint, tag=f"kp_checkbox_{keypoint}", default_value=True)

        def on_import_sleap_confirm(sender, app_data):
            selected_keypoints = [kp for kp in all_keypoints if dpg.get_value(f"kp_checkbox_{kp}")]
            process_sleap_data(all_labels, selected_keypoints)
            dpg.delete_item("sleap_keypoint_selector")
        dpg.add_button(label="Import", callback=on_import_sleap_confirm)
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        window_width = dpg.get_item_width(sleap_window)
        window_height = dpg.get_item_height(sleap_window)
        dpg.set_item_pos(sleap_window, [int((viewport_width - window_width) * 0.5), int((viewport_height - window_height) * 0.5)])

def process_sleap_data(all_labels: List[sleap_io.Labels], selected_keypoints: List[str]):
    """Processes the loaded SLEAP data and populates the sleap_annotations array."""
    global sleap_annotations
    sleap_annotations.fill(np.nan)
    for labels in all_labels:
            for frame_idx, labeled_frame in enumerate(labels):
                if frame_idx >= video_metadata['num_frames']:
                    break
                video_name = pathlib.Path(labeled_frame.video.filename).name
                try:
                    cam_idx = video_names.index(video_name)
                except ValueError:
                    continue # Video not found in the current project
                for instance in labeled_frame:
                    for node_name in selected_keypoints:
                        if node_name in [node.name for node in labels.skeleton.nodes]:
                            try:
                                point = instance[node_name]
                                x, y = point[0][0], point[0][1]
                                if not np.isnan(x) and not np.isnan(y):
                                    point_idx = POINT_NAMES.index(node_name)
                                    sleap_annotations[frame_idx, cam_idx, point_idx] = (x, y)
                            except (KeyError, ValueError):
                                continue

def toggle_ground_plane():
    """Toggles the ground plane estimation mode."""
    global ground_plane_mode
    ground_plane_mode = not ground_plane_mode
    if ground_plane_mode:
        dpg.set_value("status_message", "Select four points on the ground plane in at least two views (in the same order).")
        dpg.show_item("status_message")
        for i in range(video_metadata['num_videos']):
            ground_plane_data['points_2d'][i].clear()  # Clear previous points
    else:
        dpg.hide_item("status_message")

def calculate_sleap_annotation_counts():
    """Calculates the number of SLEAP annotations for each frame."""
    if sleap_annotations is None:
        return np.zeros(video_metadata['num_frames'])
    return np.sum(~np.isnan(sleap_annotations[:, :, :, 0]), axis=(1, 2))

def on_histogram_click(sender, app_data):
    """Callback for when the annotation histogram is clicked."""
    global frame_idx
    mouse_pos = dpg.get_plot_mouse_pos()
    if mouse_pos:
        clicked_frame = int(mouse_pos[0])
        if 0 <= clicked_frame < video_metadata['num_frames']:
            frame_idx = clicked_frame
            dpg.set_value("frame_slider", frame_idx)

def toggle_histogram(sender, app_data, user_data):
    """Toggles the visibility of the annotation histogram."""
    show = dpg.is_item_shown("histogram_window")
    if show:
        dpg.configure_item("video_grid_window", height=-1)
        dpg.hide_item("histogram_window")
    else:
        dpg.configure_item("video_grid_window", height=-150)
        dpg.show_item("histogram_window")

def toggle_manual_annotations(sender, app_data, user_data):
    global show_manual_annotations
    show_manual_annotations = not show_manual_annotations
    dpg.configure_item("annotation_histogram_series", show=app_data)

def toggle_sleap_annotations(sender, app_data, user_data):
    global show_sleap_annotations
    show_sleap_annotations = not show_sleap_annotations
    dpg.configure_item("sleap_histogram_series", show=app_data)

def toggle_mesh_roll_estimation():
    # Assumes that seed pose has first been locked to mesh axis labels
    global mesh_roll_estimation_mode, mesh_roll_points_2d, needs_3d_reconstruction, intersection_points_3d, mesh_roll_tracks
    mesh_roll_estimation_mode = not mesh_roll_estimation_mode
    mesh_roll_tracks = None # Clear previous tracks
    cam_idx = 3
    # num_frames = 20
    if mesh_roll_estimation_mode:
        if best_individual is None:
            dpg.set_value("status_message", "Error: Provide calibration before estimating mesh roll.")
            dpg.show_item("status_message")
            mesh_roll_estimation_mode = False
            return
        
        num_frames_per_iter = 100 # Run loop for 50 frames before initialising points
        start_frame = frame_idx
        # for i in range(num_frames // num_frames_per_iter):
        for i in range(1):
            idx_small, idx_large = POINT_NAMES.index('s_small'), POINT_NAMES.index('s_large')
            # frame = start_frame + (i * num_frames_per_iter)
            frame = frame_idx
            point_small_2d, point_large_2d = annotations[frame, cam_idx, idx_small], annotations[frame, cam_idx, idx_large]
            if np.any(np.isnan(point_small_2d)) or np.any(np.isnan(point_large_2d)):
                dpg.set_value("status_message", f"Error: Annotate 's_small' and 's_large' in camera {cam_idx} frame {frame} before estimating mesh roll.")
                dpg.show_item("status_message")
                mesh_roll_estimation_mode = False
                return
            mesh_roll_points_2d = scatter_pts_between(point_small_2d, point_large_2d, num_points=80)
            intersection_points_3d, intersection_mask = get_intersections_with_mesh_surface(best_individual[cam_idx], mesh_roll_points_2d, reconstructed_seed_mesh, seed_mesh_poses[frame])
            mesh_roll_points_2d = mesh_roll_points_2d[intersection_mask]
            needs_3d_reconstruction = True
            dpg.set_value("status_message", f"{len(intersection_points_3d)} intersection points found on mesh.")
            dpg.show_item("status_message")
            if len(intersection_points_3d) > 0:
                run_mesh_roll_tracking(frame, np.array(mesh_roll_points_2d), cam_idx, num_frames=num_frames_per_iter)
            else:
                mesh_roll_points_2d = [] # Clear points if no intersections
    else:
        mesh_roll_points_2d = [] # Clear points when toggling off
        intersection_points_3d = None
        needs_3d_reconstruction = True

# --- Main DPG loop ---

def main_dpg():
    """Main loop for the DearPyGui application."""
    global frame_idx, paused, needs_3d_reconstruction, best_individual, keypoint_tracking_enabled, seed_pose_tracking_enabled, focus_selected_point, save_output_video, current_frames, scene_viz
    load_videos()
    load_state()

    scene = []
    scene_viz = SceneVisualizer(frame_size=(video_metadata['width'], video_metadata['height']))
    textures = np.zeros((video_metadata['num_videos'] + 1, video_metadata['height'], video_metadata['width'], 4), dtype=np.float32) # RGBA
    create_dpg_ui(textures, scene_viz)
    # Call resize_callback once to set initial sizes
    resize_callback(None, [dpg.get_viewport_width(), dpg.get_viewport_height()], None)

    prev_frames = [None] * video_metadata['num_videos']
    prev_frame_idx = -1

    last_written_frame = -1
    num_videos = video_metadata["num_videos"] + 1 # +1 for the 3D visualization
    num_rows = num_videos // GRID_COLS + (1 if num_videos % GRID_COLS > 0 else 0)
    video_recording_buffer = np.zeros((video_metadata["height"]*num_rows, video_metadata["width"]*GRID_COLS, 3), dtype=np.uint8)

    while dpg.is_dearpygui_running():
        # Update UI text
        dpg.set_value("frame_text", f"Frame: {frame_idx}/{video_metadata['num_frames']}")
        dpg.set_value("status_text", f"Status: {'Paused' if paused else 'Playing'}")
        dpg.set_value("annotating_point_text", f"Annotating keypoint: {POINT_NAMES[selected_point_idx] if not auto_segment_mode else 'Disabled'}") # TODO: disable other functions
        dpg.set_value("focus_text", f"Focus mode: {'Enabled' if focus_selected_point else 'Disabled'}")
        dpg.set_value("fitness_text", f"Best fitness: {best_fitness_so_far:.2f}")
        dpg.set_value("num_annotations_text", f"Num annotations: {np.sum(~np.isnan(annotations[frame_idx])) // 2} / {NUM_POINTS * len(video_names)}")
        dpg.set_value("num_3d_points_text", f"Num 3D points: {np.sum(~np.isnan(reconstructed_3d_points[frame_idx]).any(axis=1))} / {NUM_POINTS}")
        dpg.set_value("tracking_text", f"Tracking: {'Enabled' if (keypoint_tracking_enabled or seed_pose_tracking_enabled) else 'Disabled'}")
        dpg.set_value("autosegment_text", f"Autosegment mode: {'Enabled' if auto_segment_mode else 'Disabled'}")
        dpg.set_value("num_calib_frames_text", f"Num calibration frames: {len(calibration_frames)}")
        dpg.set_value("save_video_text", f"Save output video: {'Enabled' if save_output_video else 'Disabled'}")
        dpg.set_value("current_frame_line", float(frame_idx))

        # Update annotation histogram based on focus mode
        if focus_selected_point:
            manual_counts = np.sum(~np.isnan(annotations[:, :, selected_point_idx, 0]), axis=1)
            sleap_counts = np.sum(~np.isnan(sleap_annotations[:, :, selected_point_idx, 0]), axis=1) if sleap_annotations is not None else np.zeros_like(manual_counts)
            dpg.configure_item("histogram_y_axis", label=f"'{POINT_NAMES[selected_point_idx]}'")
            dpg.set_axis_limits("histogram_y_axis", 0, video_metadata['num_videos'])
        else:
            manual_counts = np.sum(~np.isnan(annotations[:, :, :, 0]), axis=(1, 2))
            sleap_counts = calculate_sleap_annotation_counts()
            dpg.configure_item("histogram_y_axis", label="Total Annotations")
            max_val = max(np.max(manual_counts) if len(manual_counts) > 0 else 0, 
                          np.max(sleap_counts) if len(sleap_counts) > 0 else 0)
            dpg.set_axis_limits("histogram_y_axis", 0, max_val * 1.1)
        dpg.set_value("annotation_histogram_series", [list(range(video_metadata['num_frames'])), manual_counts.tolist()])
        dpg.set_value("sleap_histogram_series", [list(range(video_metadata['num_frames'])), sleap_counts.tolist()])

        # Frame update logic
        current_frames = []
        if prev_frame_idx != frame_idx:
            for i, cap in enumerate(video_captures):
                if prev_frame_idx != frame_idx - 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    paused = True
                    dpg.configure_item("play_pause_button", label="Play")
                    frame_idx = max(0, frame_idx - 1)
                    continue
                current_frames.append(frame)
            if keypoint_tracking_enabled and prev_frame_idx != -1 and prev_frame_idx == frame_idx - 1 and prev_frames and prev_frames[0] is not None:
                for i in range(video_metadata['num_videos']):
                    prev_gray = cv2.cvtColor(prev_frames[i], cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.cvtColor(current_frames[i], cv2.COLOR_BGR2GRAY)
                    track_points(prev_gray, gray_frame, i)
            if focus_selected_point and keypoint_tracking_enabled:
                # Check if the selected point is annotated in the current frame in at least two cameras
                count = np.sum(~np.isnan(annotations[frame_idx, :, selected_point_idx, 0]))
                if count < 2:
                    message = f"Selected keypoint '{POINT_NAMES[selected_point_idx]}' is not annotated in at least two cameras. Please annotate more points and continue."
                    dpg.set_value("status_message", message)
                    dpg.show_item("status_message")
                    paused = True
                else:
                    dpg.hide_item("status_message")
            prev_frame_idx = frame_idx
            prev_frames = current_frames
            needs_3d_reconstruction = True
        else:
            current_frames = prev_frames

        if needs_3d_reconstruction and best_individual is not None:
            needs_3d_reconstruction = False
            update_3d_reconstruction(best_individual)
            scene.clear()
            if show_cameras and show_seed_only is False:
                for i, cam in enumerate(best_individual):
                    cam_viz = create_camera_visual(cam_params=cam, scale=1.0, color=point_colors[i % len(point_colors)], label=video_names[i])
                    scene.extend(cam_viz)
            if show_seed_only is False:
                points_3d = reconstructed_3d_points[frame_idx]
                for i, point in enumerate(points_3d):
                    if not np.isnan(point).any():
                        scene.append(SceneObject(type='point', coords=point, color=point_colors[i % len(point_colors)], label=POINT_NAMES[i]))
                    from_name = POINT_NAMES[i]
                    for to in SKELETON[from_name]:
                        to_id = POINT_NAMES.index(to)
                        if not np.isnan(point).any() and not np.isnan(points_3d[to_id]).any():
                            scene.append(SceneObject(type='line', coords=np.array([point, points_3d[to_id]]), color=point_colors[i % len(point_colors)], label=None))

            if reconstructed_seed_mesh is not None and frame_idx >= initial_seed_axis_info.get('frame_idx', -1):
                points_3d = translate_rotate_mesh_3d(seed_mesh_poses[frame_idx], reconstructed_seed_mesh["points"]) # (N, 3)
                faces = reconstructed_seed_mesh["faces"]
                for face in faces:
                    p1 = points_3d[face[0]]
                    p2 = points_3d[face[1]]
                    p3 = points_3d[face[2]]
                    scene.append(SceneObject(type='line', coords=np.array([p1, p2]), color=(0, 255, 255), label=None))
                    scene.append(SceneObject(type='line', coords=np.array([p2, p3]), color=(0, 255, 255), label=None))
                    scene.append(SceneObject(type='line', coords=np.array([p3, p1]), color=(0, 255, 255), label=None))

            # # Draw the 3D intersection points for mesh roll estimation (predictions)
            # if mesh_roll_estimation_mode and predictions_3d is not None and len(predictions_3d) > 0:
            #     if frame_idx in predictions_3d.keys():
            #         for point in predictions_3d[frame_idx]:
            #             scene.append(SceneObject(type='point', coords=point, color=(0, 0, 255)))
            #         for point in expected_3d_points[frame_idx]:
            #             scene.append(SceneObject(type='point', coords=point, color=(0, 255, 255)))

            # # Draw the 3D intersection points for mesh roll estimation (queries)
            # if mesh_roll_estimation_mode and intersection_points_3d is not None and len(intersection_points_3d) > 0:
            #     show = True
            #     if mesh_roll_tracks is not None:
            #         start_frame = mesh_roll_tracks['start_frame']
            #         if (frame_idx - start_frame) >= 1:
            #             show = False
            #     if show:
            #         for point in intersection_points_3d:
            #             scene.append(SceneObject(type='point', coords=point, color=(0, 255, 0)))

            if ground_plane_data and ground_plane_data['plane_model'] and show_seed_only is False:
                draw_ground_plane(scene)

        # Update video textures
        if current_frames:
            for i, frame in enumerate(current_frames):
                frame_with_ui = draw_ui(frame.copy(), i) # (H, W, 3)
                # Convert BGR to RGBA, then to float32 and normalize
                rgba_frame = cv2.cvtColor(frame_with_ui, cv2.COLOR_BGR2RGBA) # (H, W, 4)
                textures[i, :, :, :] = rgba_frame.astype(np.float32) / 255.0
                dpg.set_value(f"video_texture_{i}", textures[i].ravel())
                # Place the frame in the recording buffer
                row = i // GRID_COLS
                col = i % GRID_COLS
                y_start = row * video_metadata['height']
                x_start = col * video_metadata['width']
                video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = frame_with_ui

        # Update 3D projection texture
        viz_3d_frame = scene_viz.draw_scene(scene)
        rgba_3d_frame = cv2.cvtColor(viz_3d_frame, cv2.COLOR_BGR2RGBA)
        textures[-1, :, :, :] = rgba_3d_frame.astype(np.float32) / 255.0
        dpg.set_value("3d_texture", textures[-1].ravel())

        # Place the 3D visualization in the recording buffer
        row = (num_videos-1) // GRID_COLS
        col = (num_videos-1) % GRID_COLS
        y_start = row * video_metadata['height']
        x_start = col * video_metadata['width']
        video_recording_buffer[y_start:y_start + video_metadata['height'], x_start:x_start + video_metadata['width']] = viz_3d_frame
        if save_output_video and video_save_output is not None and last_written_frame != frame_idx:
            video_save_output.write(video_recording_buffer)
            last_written_frame = frame_idx

        if train_ga:
            run_genetic_step()
            needs_3d_reconstruction = True

        if not paused:
            if frame_idx < video_metadata['num_frames'] - 1:
                frame_idx += 1
            else:
                paused = True
                dpg.configure_item("play_pause_button", label="Play")
            dpg.set_value("frame_slider", frame_idx)

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    for cap in video_captures:
        cap.release()
    if video_save_output is not None:
        video_save_output.release()

if __name__ == '__main__':
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir(parents=True)
        print(f"Created '{DATA_FOLDER}' directory. Please add your videos there.")
    main_dpg()