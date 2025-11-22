"""Helper functions relating to the 3D (seed) mesh segmentation and roll estimation."""
import numpy as np
import cv2
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
from scipy.optimize import minimize_scalar
from typing import Tuple

from calibration import reproject_points, undistort_points

def get_sam_segmentation(frame, point_prompt, sam_predictor):
    """Use a SAM model to segment an object based on a single point prompt."""
    if sam_predictor is None:
        print("SAM model is not initialized.")
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb_frame)
    input_point = np.array([point_prompt])
    input_label = np.array([1]) # 1 indicates a foreground point
    masks, scores, logits = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False, # We want the single best mask
    )
    if masks is None or len(masks) == 0:
        return None
    mask = masks[0].astype(np.uint8) # Convert the boolean mask to contour points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour.squeeze(axis=1) # Shape: (N, 2)

def translate_rotate_mesh_3d(pose: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """Translate and rotate the seed mesh points based on a given seed mesh pose."""
    # pose (6,) tx, ty, tz, rx, ry, rz
    tvec = pose[:3]
    rvec = pose[3:]
    R, _ = cv2.Rodrigues(rvec)
    transformed_points = (R @ points_3d.T).T + tvec
    return transformed_points  # (points, 3)

def reproject_mesh_segmentation_as_contour(cam_idx: int, pose: np.ndarray, reconstructed_mesh, best_individual) -> np.ndarray:
    """Reproject 2d seed contour (shadow) from 3d reconstructed mesh."""
    if reconstructed_mesh is None or best_individual is None:
        return
    points_3d = translate_rotate_mesh_3d(pose, reconstructed_mesh["points"]) # (points, 3)
    vertex_points_3d = points_3d[reconstructed_mesh['vertices']] # (vertices, 3)
    reprojected = reproject_points(vertex_points_3d, best_individual[cam_idx]) # (vertices, 2)
    hull = ConvexHull(reprojected)
    hull = reprojected[hull.vertices]  # (hull_points, 2)
    return hull

def resample_contour(contour: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a 2D contour to have a specific number of points."""
    closed_contour = np.vstack([contour, contour[0]])
    distances = np.cumsum(np.sqrt(np.sum(np.diff(closed_contour, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    total_length = distances[-1]
    if total_length < 1e-6: # Handle case of single or identical points
        return np.repeat(contour[:1], num_points, axis=0)
    resampled_distances = np.linspace(0, total_length, num_points, endpoint=False) # Use endpoint=False for closed loop
    resampled_x = np.interp(resampled_distances, distances, closed_contour[:, 0])
    resampled_y = np.interp(resampled_distances, distances, closed_contour[:, 1])
    return np.vstack((resampled_x, resampled_y)).T

def calculate_mesh_poses_from_axis(pt_names, axis_names, initial_axis_info, num_frames, mesh_poses, points_3d):
    """Calculates the pose of the seed mesh for every frame, by its alignment with the
    's_small' and 's_large' keypoints.
    """
    if not initial_axis_info:
        print("Error: Initial seed axis information not available.")
        return
    s_small_idx = pt_names.index(axis_names[0])
    s_large_idx = pt_names.index(axis_names[1])
    # Get initial axis information from the frame where the mesh was reconstructed
    initial_frame_idx = initial_axis_info['frame_idx']
    initial_s_small_3d = initial_axis_info[f'{axis_names[0]}_3d']
    initial_s_large_3d = initial_axis_info[f'{axis_names[1]}_3d']

    if np.isnan(initial_s_small_3d).any() or np.isnan(initial_s_large_3d).any():
        print("Error: The initial seed axis points were not annotated in the reconstruction frame.")
        return

    # Change seed mesh pose according to rotation and translation changes of seed
    initial_midpoint = (initial_s_small_3d + initial_s_large_3d) / 2
    initial_axis_vec = initial_s_large_3d - initial_s_small_3d
    initial_axis_vec_norm = initial_axis_vec / np.linalg.norm(initial_axis_vec)  # Norm axis vectors
    for f in range(initial_frame_idx, num_frames):
        current_s_small_3d = points_3d[f, s_small_idx]
        current_s_large_3d = points_3d[f, s_large_idx]
        # If axis points are not available for the current frame, extrapolate from the previous frame
        if np.isnan(current_s_small_3d).any() or np.isnan(current_s_large_3d).any():
            if f > 0:
                mesh_poses[f] = mesh_poses[f - 1] # Use pose from the previous frame
            continue
        current_midpoint = (current_s_small_3d + current_s_large_3d) / 2
        current_axis_vec = current_s_large_3d - current_s_small_3d
        current_axis_vec_norm = current_axis_vec / np.linalg.norm(current_axis_vec)
        # Find the rotation that aligns the original axis vector with the current one
        rotation, _ = Rotation.align_vectors(current_axis_vec_norm, initial_axis_vec_norm)
        rotation_matrix = rotation.as_matrix()
        rvec, _ = cv2.Rodrigues(rotation_matrix)
        # Calculate the translation
        tvec = current_midpoint - (rotation_matrix @ initial_midpoint)
        mesh_poses[f, :3] = tvec.flatten()
        mesh_poses[f, 3:] = rvec.flatten()
    return mesh_poses

def scatter_pts_between(pt1_2d, pt2_2d, num_points=40):
    centre = (pt1_2d + pt2_2d) / 2.0
    # Major and minor axes for the covariance
    vec_major = pt2_2d - pt1_2d
    len_major = np.linalg.norm(vec_major)
    std_dev_major = len_major / 4.0  # SD along the major axis, points within +/- 2-sigma
    std_dev_minor = std_dev_major / 3.0  # Fraction of major axis to create an oval
    # Rotation angle of major axis with respect to x-axis
    angle = np.arctan2(vec_major[1], vec_major[0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], 
                                [sin_a, cos_a]])
    # Un-rotated covariance matrix (aligned with x/y axes)
    cov_unrotated = np.array([[std_dev_major**2, 0], 
                            [0, std_dev_minor**2]])
    cov_rotated = rotation_matrix @ cov_unrotated @ rotation_matrix.T
    return np.random.multivariate_normal(centre, cov_rotated, num_points)

def get_intersections_with_mesh_surface(cam_params, points_2d, mesh, mesh_pose):
    """Return 3d coordinates of points of intersection if world space."""
    num_points = points_2d.shape[0]
    rvec = cam_params['rvec']
    tvec = cam_params['tvec']
    R, _ = cv2.Rodrigues(rvec) # Convert rvec to 3x3 R matrix
    ext_matrix_world_to_cam = np.eye(4)
    ext_matrix_world_to_cam[:3, :3] = R
    ext_matrix_world_to_cam[:3, 3] = tvec.flatten()
    ext_matrix_cam_to_world = np.linalg.inv(ext_matrix_world_to_cam)
    R_cam_to_world = ext_matrix_cam_to_world[:3, :3]
    ray_origin_world = ext_matrix_cam_to_world[:3, 3]
    ray_origins_world = np.tile(ray_origin_world, (num_points, 1))
    transformed_vertices = translate_rotate_mesh_3d(
        mesh_pose, 
        mesh["points"]
    )
    mesh_to_intersect = trimesh.Trimesh(
        vertices=transformed_vertices,
        faces=mesh["faces"]
    )
    intersector = mesh_to_intersect.ray
    undistorted_pixels = undistort_points(points_2d, cam_params)
    fx, fy, cx, cy = cam_params['fx'], cam_params['fy'], cam_params['cx'], cam_params['cy']
    u = undistorted_pixels[:, 0]
    v = undistorted_pixels[:, 1]
    x_cam, y_cam = (u - cx) / fx, (v - cy) / fy
    ray_dirs_cam = np.vstack([x_cam, y_cam, np.ones(num_points)]).T  # (x, y, 1.0)
    ray_dirs_cam /= np.linalg.norm(ray_dirs_cam, axis=1, keepdims=True)
    ray_dirs_world = (R_cam_to_world @ ray_dirs_cam.T).T  # Transform ray dir to world space
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins=ray_origins_world,
        ray_directions=ray_dirs_world
    )
    unique_ray_indices = np.unique(index_ray)

    intersection_mask = np.zeros(num_points, dtype=bool)  # To indicate which of the rays have intersection
    intersection_mask[unique_ray_indices] = True
    entry_points = []
    for ray_idx in unique_ray_indices:
        hit_mask = (index_ray == ray_idx)
        ray_locations = locations[hit_mask]
        ray_origin = ray_origins_world[ray_idx]
        distances_sq = np.sum((ray_locations - ray_origin)**2, axis=1)
        entry_point = ray_locations[np.argmin(distances_sq)]
        entry_points.append(entry_point) 
    return np.array(entry_points), intersection_mask # (N, 3), (M,)

def pose_vec_to_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6-element pose vector (3-elem Rodrigues rotation + 3-elem translation)
    into a 4x4 homogeneous transformation matrix.
    """
    # Extract rotation (Rodrigues vector) and translation
    rod_vec = pose_vec[:3]
    t_vec = pose_vec[3:].reshape(3, 1)  # Ensure translation is a column vector
    # Convert Rodrigues vector to 3x3 rotation matrix using cv2
    R, _ = cv2.Rodrigues(rod_vec)
    # Create 4x4 homogeneous transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R          # Set rotation part
    transform_matrix[:3, 3] = t_vec.flatten()  # Set translation part
    return transform_matrix

def calculate_change_between_poses(pose1: np.ndarray, pose2: np.ndarray):
    rvec_A = pose1[:3]
    tvec_A = pose1[3:].reshape(3, 1) # Reshape to 3x1 column vector
    rvec_B = pose2[:3]
    tvec_B = pose2[3:].reshape(3, 1) # Reshape to 3x1 column vector
    R_A, _ = cv2.Rodrigues(rvec_A)
    R_B, _ = cv2.Rodrigues(rvec_B)
    R_change = R_B @ R_A.T  # Change in rotation
    t_change = tvec_B - (R_change @ tvec_A)
    rvec_change, _ = cv2.Rodrigues(R_change)
    tvec_change_flat = t_change.flatten()
    rvec_change_flat = rvec_change.flatten()    
    return np.hstack((rvec_change_flat, tvec_change_flat))


def find_optimal_roll(
    source_points: np.ndarray,
    target_points: np.ndarray,
    axis_vector_norm: np.ndarray,
    axis_center: np.ndarray
) -> Tuple[float, float]:
    """
    Finds optimal roll required to match source points to target points,
    around a vector axis.
    """
    def objective_func(theta: float) -> float:
        rotation = Rotation.from_rotvec(theta * axis_vector_norm)
        source_translated = source_points - axis_center
        source_rotated = rotation.apply(source_translated)
        source_rolled = source_rotated + axis_center
        return np.sum((source_rolled - target_points)**2)
    result = minimize_scalar(
        objective_func,
        bounds=(-np.pi, np.pi),
        method='bounded'
    )
    return result.x, result.fun

# def estimate_mesh_pose():
#     """Estimates the seed mesh pose for the current frame using PnP."""
#     if reconstructed_seed_mesh is None or best_individual is None:
#         return
#     def calc_error(pose: np.ndarray) -> float:
#         cameras_with_seed_segmentation = [i for i, seeds in enumerate(seed_points_2d) if len(seeds) > 0]
#         total_error = 0
#         count = 0
#         for i in cameras_with_seed_segmentation:
#             contour_2d = reproject_mesh_segmentation_as_contour(i, pose, reconstructed_seed_mesh, best_individual)
#             if contour_2d is None or len(contour_2d) < 5:
#                 continue
#             seed_2d = np.array(seed_points_2d[i], dtype=np.float32)
#             if len(seed_2d) < 5:
#                 continue
#             # Resample both contours to a fixed number of points
#             resampled_projected = resample_contour(contour_2d, 100)
#             resampled_observed = resample_contour(seed_2d, 100)
#             # Find the best alignment by shifting the starting point
#             best_error_for_cam = float('inf')
#             for offset in range(len(resampled_projected)):
#                 rolled_contour = np.roll(resampled_projected, offset, axis=0)
#                 pointwise_distances = np.linalg.norm(resampled_observed - rolled_contour, axis=1)
#                 # Use the 90th percentile instead of the mean to ignore the worst 10% of errors (outliers)
#                 error = np.percentile(pointwise_distances, 90)
#                 if error < best_error_for_cam:
#                     best_error_for_cam = error
#             total_error += best_error_for_cam
#             count += 1
#         print("Total robust seed reprojection error:", total_error, "Count:", count, "pose", pose)
#         return total_error / count if count > 0 else float('inf')
#     # Initial guess for pose
#     initial_pose = seed_mesh_poses[frame_idx]
#     print("Optimizing seed mesh pose for frame", frame_idx)
#     result = minimize(calc_error, initial_pose, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-2, 'fatol': 1e-2, 'disp': False})
#     print("Result", result)
#     if result.success:
#         seed_mesh_poses[frame_idx] = result.x
#     else:
#         print("Seed mesh pose optimisation failed for frame", frame_idx)