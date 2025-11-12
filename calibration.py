"""Helper functions relating to the camera calibration and parameter estimation."""
from typing import TypedDict, List
import numpy as np
import cv2
import itertools
import random

NUM_DIST_COEFFS = 14  # Number of distortion coefficients
class CameraParams(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray

def flat_camera_params(cam: CameraParams) -> np.ndarray:
    """Returns a flattened array of camera parameters for genetic algorithm."""
    """Flattens the CameraParams into a single array."""
    return np.concatenate([
        np.array([cam['fx'], cam['fy'], cam['cx'], cam['cy']], dtype=np.float32),
        cam['dist'][:NUM_DIST_COEFFS].astype(np.float32),
        cam['rvec'].astype(np.float32),
        cam['tvec'].astype(np.float32)
    ])

def unflat_camera_params(cam: CameraParams, flattened: np.ndarray):
    """Creates a CameraParams object from a flattened array."""
    """Unflattens a single array into CameraParams."""
    cam['fx'] = flattened[0]
    cam['fy'] = flattened[1]
    cam['cx'] = flattened[2]
    cam['cy'] = flattened[3]
    cam['dist'] = flattened[4:4 + NUM_DIST_COEFFS].astype(np.float32)
    cam['rvec'] = flattened[4 + NUM_DIST_COEFFS:4 + NUM_DIST_COEFFS + 3].astype(np.float32)
    cam['tvec'] = flattened[4 + NUM_DIST_COEFFS + 3:4 + NUM_DIST_COEFFS + 6].astype(np.float32)

def flat_individual(individual: List[CameraParams]) -> np.ndarray:
    """Flattens a list of CameraParams into a single array."""
    return np.concatenate([flat_camera_params(cam) for cam in individual])

def unflat_individual(flattened: np.ndarray, num_cameras: int) -> List[CameraParams]:
    """Unflattens a single array into a list of CameraParams."""
    cam_params_list = []
    offset = 0
    for _ in range(num_cameras):
        cam_params = CameraParams(
            fx=flattened[offset],
            fy=flattened[offset + 1],
            cx=flattened[offset + 2],
            cy=flattened[offset + 3],
            dist=flattened[offset + 4:offset + 4 + NUM_DIST_COEFFS],
            rvec=flattened[offset + 4 + NUM_DIST_COEFFS:offset + 4 + NUM_DIST_COEFFS + 3],
            tvec=flattened[offset + 4 + NUM_DIST_COEFFS + 3:offset + 4 + NUM_DIST_COEFFS + 6]
        )
        cam_params_list.append(cam_params)
        offset += (4 + NUM_DIST_COEFFS + 6)  # Move to the next camera's parameters
    return cam_params_list

def get_camera_matrix(cam_params: CameraParams) -> np.ndarray:
    """Constructs the camera matrix from camera parameters."""
    K = np.array([[cam_params['fx'], 0, cam_params['cx']],
                  [0, cam_params['fy'], cam_params['cy']],
                  [0, 0, 1]], dtype=np.float32)
    return K # (3, 3)

def get_projection_matrix(cam_params: CameraParams) -> np.ndarray:
    """Constructs the projection matrix from camera parameters."""
    K = get_camera_matrix(cam_params)
    R, _ = cv2.Rodrigues(cam_params['rvec'])
    return K @ np.hstack((R, cam_params['tvec'].reshape(-1, 1))) # (3, 4)

def undistort_points(points_2d: np.ndarray, cam_params: CameraParams) -> np.ndarray:
    """Undistorts 2D points using camera parameters."""
    # points_2d: (..., 2)
    valid_points_mask = ~np.isnan(points_2d).any(axis=-1)  # Mask for valid points
    if not np.any(valid_points_mask):
        return np.full_like(points_2d, np.nan, dtype=np.float32)
    valid_points = points_2d[valid_points_mask]  # Extract valid points (num_valid_points, 2)
    undistorted_full = np.full_like(points_2d, np.nan, dtype=np.float32)  # Prepare output array with NaNs
    # The following function returns normalised coordinates, not pixel coordinates
    undistorted_points = cv2.undistortImagePoints(valid_points.reshape(-1, 1, 2), get_camera_matrix(cam_params), cam_params['dist']) # (num_valid_points, 1, 2)
    undistorted_full[valid_points_mask] = undistorted_points.reshape(-1, 2)  # Fill only valid points
    return undistorted_full  # Shape: (..., 2) with NaNs for invalid points

def combination_triangulate(frame_annotations: np.ndarray, proj_matrices: np.ndarray) -> np.ndarray:
    """Triangulates 3D points from 2D correspondences using multiple camera views."""
    # frame_annotations: (num_frames, num_cams, num_points, 2) and proj_matrices: (num_cams, 3, 4)
    # returns (points_3d: (num_frames, num_points, 3))
    assert frame_annotations.shape[1] == proj_matrices.shape[0], "Number of cameras must match annotations."
    combs = list(itertools.combinations(range(proj_matrices.shape[0]), 2))
    # Every combination makes a prediction, some combinations may not have enough points to triangulate
    points_3d = np.full((frame_annotations.shape[0], len(combs), frame_annotations.shape[2], 3), np.nan, dtype=np.float32)  # (num_frames, num_combs, num_points, 3)
    for idx, (i, j) in enumerate(combs):
        # Get 2D points from both cameras
        p1_2d = frame_annotations[:, i] # (num_frames, num_points, 2)
        p2_2d = frame_annotations[:, j] # (num_frames, num_points, 2)
        common_mask = ~np.isnan(p1_2d).any(axis=2) & ~np.isnan(p2_2d).any(axis=2)  # (num_frames, num_points,)
        if not np.any(common_mask):
            continue
        # Prepare 2D points for triangulation (requires shape [2, N])
        p1_2d = p1_2d[common_mask] # (num_common_points, 2)
        p2_2d = p2_2d[common_mask] # (num_common_points, 2)
        # Expects (3, 4) project matrices and (2, N) points
        points_4d_hom = cv2.triangulatePoints(proj_matrices[i], proj_matrices[j], p1_2d.T, p2_2d.T) # (4, num_common_points) homogenous coordinates
        triangulated_3d = (points_4d_hom[:3] / points_4d_hom[3]).T  # Convert to 3D coordinates (num_common_points, 3)
        points_3d[:, idx][common_mask] = triangulated_3d
    # Average the triangulated points across all combinations
    average = np.nanmean(points_3d, axis=1)  # (num_frames, num_points, 3)
    return average

def reproject_points(points_3d: np.ndarray, cam_params: CameraParams) -> np.ndarray:
    """Reprojects 3D points back to 2D image plane using camera parameters."""
    # points_3d: (N, 3)
    # cam_params: CameraParams
    points_3d = points_3d.reshape(-1, 1, 3)  # Shape: (N, 1, 3)
    reprojected_pts_2d, _ = cv2.projectPoints(
        points_3d, cam_params['rvec'], cam_params['tvec'], get_camera_matrix(cam_params), cam_params['dist']
    )  # (N, 1, 2)
    if reprojected_pts_2d is None:
        raise ValueError("Reprojection failed, check camera parameters and 3D points.")
    return reprojected_pts_2d.squeeze(axis=1)  # Shape: (N, 2)

def create_individual(video_metadata) -> List[CameraParams]:
    """Creates a single individual with a robust lookAt orientation."""
    num_cameras = video_metadata['num_videos']
    w, h = video_metadata['width'], video_metadata['height']
    radius = 5  # Initial guess for camera distance from origin
    individual = []
    for i in range(num_cameras):
        # Intrinsics
        fx = random.uniform(w * 0.8, w * 1.5)
        fy = random.uniform(h * 0.8, h * 1.5)
        cx = w / 2 + random.uniform(-w * 0.05, w * 0.05)
        cy = h / 2 + random.uniform(-h * 0.05, h * 0.05)
        # Distortion (keep it small initially)
        dist = np.random.normal(0.0, 0.001, size=NUM_DIST_COEFFS).astype(np.float32)
        # Calculate tvec: position the camera in a circle
        angle = (2 * np.pi / num_cameras) * i
        x = radius * np.cos(angle) + random.uniform(-0.1, 0.1)
        y = 2 + random.uniform(-0.5, 0.5)
        z = radius * np.sin(angle) + random.uniform(-0.1, 0.1)
        cam_in_world = np.array([x, y, z], dtype=np.float32)
        # Calculate rvec: make the camera "look at" the target
        # Forward vector (from camera to target)
        # The point all cameras are looking at
        target = np.array([0, 0, 0], dtype=np.float32)
        world_up = np.array([0, 1, 0], dtype=np.float32)
        forward = (target - cam_in_world) / np.linalg.norm(target - cam_in_world)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        cap_up = np.cross(forward, right)

        R = np.array([right, cap_up, forward])
        rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to rotation vector
        tvec = -R @ cam_in_world  # Translation vector to move the camera to the origin
        individual.append(CameraParams(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, rvec=rvec.flatten(), tvec=tvec.flatten()))
    return individual

def mutate(individual: List[CameraParams]) -> List[CameraParams]:
    """Mutates an individual by applying small random changes."""
    mutated = []
    alpha = 0.01
    for cam_params in individual:
        # Mutate intrinsics
        fx = cam_params['fx'] + np.random.normal(0, alpha)
        fy = cam_params['fy'] + np.random.normal(0, alpha)
        cx = cam_params['cx'] + np.random.normal(0, alpha)
        cy = cam_params['cy'] + np.random.normal(0, alpha)
        # Mutate distortion
        dist = cam_params['dist'] + np.random.normal(0, alpha, size=cam_params['dist'].shape[0])
        # Mutate extrinsics
        rvec = cam_params['rvec'] + np.random.normal(0, np.pi/180, size=3)
        tvec = cam_params['tvec'] + np.random.normal(0, alpha, size=3)
        mutated.append(CameraParams(fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, rvec=rvec, tvec=tvec))
    # Anchor camera 0 to the origin
    # mutated[0]['rvec'] = np.zeros(3, dtype=np.float32)  # No rotation
    # mutated[0]['tvec'] = np.zeros(3, dtype=np.float32)  # No translation
    return mutated

def fitness(individual: List[CameraParams], annotations: np.ndarray, calibration_frames: List[int], human_annotated: np.ndarray) -> float:
    """
    Refactored fitness function that iterates frame-by-frame and processes points in batches.

    This version calculates multiple triangulation points from pairs of cameras for a given valid frame.
    Lower reprojection error results in a higher fitness score.
    """
    reprojection_errors = []
    num_cams = annotations.shape[1]
    # Get projection matrices and camera parameters once to avoid redundant calculations in the loop.
    # We are ignoring the intrinsics because we undistort the points before triangulation.
    proj_matrices = np.array([get_projection_matrix(i) for i in individual]) # (num_cams, 3, 4)

    if len(calibration_frames) == 0:
        return float('inf')  # No calibration frames found / selected
    # Find frames with at least one valid annotation to process.
    valid_frames_mask = np.any(~np.isnan(annotations), axis=(1, 2, 3)) & np.any(human_annotated, axis=(1, 2)) # (num_frames,)
    calibration_frames_mask = np.zeros_like(valid_frames_mask, dtype=bool)  # (num_frames,)
    calibration_frames_mask[calibration_frames] = True  # Mark calibration frames as valid
    valid_frames_mask = valid_frames_mask & calibration_frames_mask  # Only consider frames that are both valid and in the calibration set
    valid_annotations = annotations[valid_frames_mask]  # (num_valid_frames, num_cams, num_points, 2)
    undistorted_annotations = np.full_like(valid_annotations, np.nan, dtype=np.float32)  # (num_valid_frames, num_cams, num_points, 2)

    for c in range(num_cams):
        undistorted_annotations[:, c] = undistort_points(valid_annotations[:, c], individual[c])  # (num_valid_frames, num_points, 2)

    points_3d = combination_triangulate(undistorted_annotations, proj_matrices) # (num_valid_frames, num_points, 3)
    valid_3d_mask = ~np.isnan(points_3d).any(axis=-1)  # (num_valid_frames, num_points)
    for c in range(num_cams):
        # Reproject 3d points back to 2d for this camera
        valid_2d_mask = ~np.isnan(valid_annotations[:, c]).any(axis=-1)  # (num_valid_frames, num_points)
        common_mask = valid_3d_mask & valid_2d_mask  # Points that are valid in both 3D and 2D
        valid_3d_points = points_3d[common_mask]  # (num_common_points, 3)
        valid_2d_points = valid_annotations[:, c][common_mask]  # (num_common_points, 2)
        reprojected = reproject_points(valid_3d_points, individual[c]) # (num_common_points, 2)
        # Calculate the reprojection error for valid points
        error = np.linalg.norm(reprojected - valid_2d_points, axis=1)  # Euclidean distance, (num_common_points,)
        reprojection_errors.extend(error)  # Append the error for this camera

    if len(reprojection_errors) == 0:
        return float('inf')  # No valid points to evaluate

    # average_error = total_reprojection_error / points_evaluated
    average_error = np.sum(reprojection_errors)
    # print("Descriptive statistics of reprojection errors:")
    # print(f"  Min: {np.min(reprojection_errors):.2f}, Max: {np.max(reprojection_errors):.2f}, Mean: {average_error:.2f}, Std Dev: {np.std(reprojection_errors):.2f}")
    return average_error  # Fitness is the error

def permutation_optimisation(individual: List[CameraParams], annotations: np.ndarray, calibration_frames: List[int], human_annotated: np.ndarray):
    """Optimise the order of cameras in the individual to minimise reprojection error."""
    # Evaluate the fitness of the current individual for every permutation
    perms = list(itertools.permutations(individual))
    fitness_scores = np.array([fitness(list(p), annotations, calibration_frames, human_annotated) for p in perms]) # (num_permutations,)
    best_perm = perms[np.argmin(fitness_scores)]
    individual[:] = list(best_perm)  # Update the individual with the best permutation

def calculate_all_reprojection_errors(video_metadata, video_names, point_names, best_individual, annotations, reconstructed_3d_points) -> List[dict]:
    """Finds the top_k worst reprojection errors across all frames, cameras, and points."""
    if 'best_individual' not in globals() or best_individual is None:
        print("Optimized camera parameters ('best_individual') not found.")
        return []
    # Compute reconstruction
    undistorted_annotations = np.full_like(annotations, np.nan, dtype=np.float32)  # (frames, num_cams, num_points, 2)
    for c in range(video_metadata['num_videos']):
        undistorted_annotations[:, c] = undistort_points(annotations[:, c], best_individual[c])  # (frames, num_points, 2)
    proj_matrices = np.array([get_projection_matrix(i) for i in best_individual])
    points_3d = combination_triangulate(undistorted_annotations, proj_matrices) # (frames, num_points, 3)
    reconstructed_3d_points[:] = points_3d  # Update the global 3D points for this frame

    all_errors = []
    # Pre-calculate masks for valid 3D points and 2D annotations to avoid re-computation
    valid_3d_mask = ~np.isnan(reconstructed_3d_points).any(axis=-1)  # Shape: (num_frames, num_points)
    valid_2d_mask = ~np.isnan(annotations).any(axis=-1)  # Shape: (num_frames, num_cams, num_points)

    # Iterate over each camera to calculate its reprojection errors
    for cam_idx in range(video_metadata['num_videos']):
        # Find points that are valid in both the 3D data and this camera's 2D annotations
        common_mask = valid_3d_mask & valid_2d_mask[:, cam_idx]
        # Get the (frame_idx, point_idx) coordinates for all valid points
        frame_indices, point_indices = np.where(common_mask)
        # If no valid points exist for this camera, skip to the next one
        if frame_indices.size == 0:
            continue

        # Select the corresponding 3D points and 2D ground truth annotations
        points_3d = reconstructed_3d_points[frame_indices, point_indices]
        points_2d = annotations[frame_indices, cam_idx, point_indices]
        # Reproject the 3D points onto the current camera's 2D image plane
        reprojected_points = reproject_points(points_3d, best_individual[cam_idx]) # (num_valid_points, 2)

        valid_reprojection_mask = (reprojected_points[:, 0] >= 0) & (reprojected_points[:, 0] < video_metadata['width']) & \
                                  (reprojected_points[:, 1] >= 0) & (reprojected_points[:, 1] < video_metadata['height'])
        # Filter out points that are outside the image bounds
        reprojected_points = reprojected_points[valid_reprojection_mask]
        points_2d = points_2d[valid_reprojection_mask]
        frame_indices = frame_indices[valid_reprojection_mask]
        point_indices = point_indices[valid_reprojection_mask]
        # Calculate the Euclidean distance (L2 norm) between reprojected and annotated points
        errors = np.square(reprojected_points - points_2d)
        errors = np.sqrt(np.sum(errors, axis=-1))  # Shape: (num_valid_points,)
        # Store each error along with its full context (frame, camera, point index)
        for i in range(len(errors)):
            all_errors.append({
                'error': float(errors[i]),
                'frame': int(frame_indices[i]),
                'camera': video_names[cam_idx],
                'point': point_names[point_indices[i]],
                'annotated_point': annotations[frame_indices[i], cam_idx, point_indices[i]].tolist(),
                'reprojected_point': reprojected_points[i].tolist(),
                '3d_point': points_3d[i].tolist()
            })

    # Sort the collected errors in descending order to find the largest ones
    sorted_errors = sorted(all_errors, key=lambda x: x['error'], reverse=True)
    return sorted_errors