from typing import List, Tuple, Dict
import numpy as np


def linear_regression(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Fit a linear regression model to a set of points.

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

    return [a, b]


def predict_trajectory_vector(
    id_dict: Dict[str, List[Tuple[float, float]]], threshold: int = 5
) -> Tuple[Dict[str, List[Tuple[float, float]]], np.ndarray]:
    """
    Predicts the trajectory vector for each object ID in the given dictionary.

    Args:
        id_dict (Dict[List[Tuple[float, float]]]): A dictionary containing object IDs as keys and a list of trajectory points as values.
        threshold: The minimum number of trajectory points required for an object ID to be considered valid.

    Returns:
        Tuple[Dict[List[Tuple[float, float]]], np.ndarray]: A tuple containing two elements:
            - valid_dict: A dictionary containing only the valid object IDs and their corresponding trajectory points.
            - vector_dict: A NumPy array containing the trajectory vectors for each valid object ID.
    """

    valid_dict = {}
    trajectory_vectors = []

    for obj_id, points in sorted(id_dict.items()):
        if len(points) >= threshold:
            valid_dict[obj_id] = points
            trajectory_vectors.append(linear_regression(points))

    return valid_dict, np.array(trajectory_vectors)
