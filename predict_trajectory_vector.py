from typing import List, Tuple, Dict
import numpy as np


def linear_regression(points: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Fit a linear regression model to a set of points.
    y = ax + b

    Args:
    - points: a list of (x, y) coordinates

    Returns:
    - a numpy array representing the coefficients of the linear regression model
    """
    points_arr = np.array(points)
    x = points_arr[:, 0]
    y = points_arr[:, 1]

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.dot(x, x)
    sum_xy = np.dot(x, y)

    # Linear regression
    b = (sum_y * sum_x2 - sum_xy * sum_x) / (len(x) * sum_x2 - sum_x**2)
    a = sum_xy / sum_x2 - b * (sum_x / sum_x2)

    # modified by duy, check logic
    current_trajectory = (int(x[-1] - x[0]), int(a*(x[-1] - x[0])))

    # simulate future trajectory by 100 points or if either x, or y reach the border of the image
    # assuming it the resolution where

    max_x = 1920
    max_y = 1080
    x_dir = current_trajectory[0]
    y_dir = current_trajectory[1]
    magnitude = np.sqrt(x_dir**2 + y_dir**2)

    length_traj = 400 # scale the pred_trajectory to max 5 pixels in length
    if magnitude > length_traj:
        scale_factor = length_traj/magnitude
        x_dir *= scale_factor
        y_dir *= scale_factor

    pred_x = np.clip(x[-1] + x_dir, a_min=0, a_max=max_x-1)
    pred_y = np.clip(y[-1] + y_dir, a_min=0, a_max=max_y-1)
    pred_trajectory = (round(pred_x), round(pred_y))


    # x_ahead = int(x[-1] + x_inc)
    # y_pred = int(a * x_ahead + b)

    # while (x_ahead < max_x-1) and (y_pred < max_y-1) and len(pred_traj) < 15:
    #     pred_traj.append((x_ahead, y_pred))
    #     x_ahead = int(x_ahead + x_inc)
    #     y_pred = int(a * x_ahead + b)

    return current_trajectory, pred_trajectory


def predict_trajectory_vector(
    id_dict: Dict[str, List[Tuple[float, float, float, float]]], threshold: int = 5
) -> Tuple[Dict[str, List[Tuple[float, float, float, float]]], np.ndarray]:
    """
    Predicts the trajectory vector for each object ID in the given dictionary.

    Args:
        id_dict (Dict[List[Tuple[float, float]]]): A dictionary containing object IDs as keys and a list of trajectory points as values.
        threshold: The minimum number of trajectory points required for an object ID to be considered valid.

    Returns:
        Tuple[Dict[List[Tuple[float, float]]], np.ndarray]: A tuple containing two elements:
            - valid_dict: A dictionary containing only the valid object IDs and their corresponding trajectory points.
            - vector_dict: A NumPy array containing the current trajectory vectors for each valid object ID.
            - vector_dict: A NumPy array containing the predicted tracking points of the vectors for each valid object ID, size (n, 100, 2)
    """

    filtered_idx = []
    locs = []
    cur_trajs = []
    pred_trajs = []

    for obj_idx, points in sorted(id_dict.items()):
        if len(points) >= threshold:
            filtered_idx.append(obj_idx)
            locs.append((round(points[-1][0]), round(points[-1][1]))) # only return last location
            current_trajectory, pred_traj = linear_regression(points)
            cur_trajs.append(current_trajectory)
            pred_trajs.append(pred_traj)

    return np.array(locs), np.array(cur_trajs), np.array(pred_trajs), filtered_idx
