import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_attraction_matrix_from_boxes(per_boxes, obj_boxes):
    """Assume per_boxes and obj_boxes is a dict with key is id, and value is boxes (x,y,w,h).
    """

    per_loc = np.array([box[:2] for box in per_boxes.values()])
    obj_loc = np.array([box[:2] for box in obj_boxes.values()])
    return get_attraction_matrix(per_loc, obj_loc)

def get_attraction_matrix(per_loc, obj_loc):
    """
        Matrix: each value is a vector from a person to an object 
        Args:
            per_loc [p, 2]: location of p persons of interest
            obj_loc [o, 2]: location of o objects of interest
            Each row is a vector of location with coor [x, y]
        Return:
            per_obj_attract [p, o, 2]: Matrix of vector from location of each person to each object
    """
    
    diff_matrix = obj_loc[:, None, :] - per_loc[:, :]
    diff_matrix = np.transpose(diff_matrix, (1, 0, 2))
    return diff_matrix

def get_similarity_vector_matrix(attraction_matrix, per_traj):
    """
        Matrix: each value is a vector from a person to an object 
        Args:
            attraction_matrix [p, o, 2]: Matrix of vector from location of each person to each object,
                retrieved from get_attraction_matrix()
            per_traj [p, 2]: trajectory of p persons of interest,
                retrieved from predict_trajectory_vector()
        Return:
            per_obj_sim [p, o]: Matrix of similarity bw direction and actual trajectory of each person to each object
    """

    per_traj_reshaped = per_traj[:, None, :]
    dot_product = np.sum(attraction_matrix * per_traj_reshaped, axis=-1)
    norm_v1 = np.linalg.norm(attraction_matrix, axis=-1)
    norm_v2 = np.linalg.norm(per_traj_reshaped, axis=-1)
    per_obj_sim = np.divide(dot_product, (norm_v1 * norm_v2))
    
    return per_obj_sim


if __name__ == "__main__":
    per_loc = np.array([[1, 2], [3, 4], [-1, -2], [-3, -4]])    # 4 persons
    obj_loc = np.array([[7, 8], [9, 10], [11, 12]])             # 3 objects
    exp_res = np.array(
        [
            [
                [6, 6],
                [8, 8],
                [10, 10],
            ],
            [
                [4, 4],
                [6, 6],
                [8, 8],
            ],
            [
                [8, 10],
                [10, 12],
                [12, 14],
            ],
            [
                [10, 12],
                [12, 14],
                [14, 16]
            ]
        ]
    )
    # print("per_traj", per_loc.shape, "\n", per_loc, "\n")
    # print("obj_tracj", obj_loc.shape, "\n", obj_loc, "\n")

    per_obj_attract = get_attraction_matrix(per_loc, obj_loc)
    equal = np.array_equal(per_obj_attract, exp_res)
    # print("Out come is closed to expected value:", equal)
    if not equal:
        print(per_obj_attract)


    per_traj = np.array([[9.3, 3.4], [-3.9, 0.4], [-1.3, 4.2], [3.3, -4.9]])
    sim_vec_matrix = get_similarity_vector_matrix(per_obj_attract, per_traj)

    per_id = 1
    obj_id = 2
    ran_vec = per_obj_attract[per_id, obj_id,:]
    ran_per = per_traj[per_id, :]

    print(ran_vec[None,:].shape)
    print(ran_per[None,:].shape)

    cos_sim = cosine_similarity(ran_vec[None,:], ran_per[None,:])
    print(cos_sim, sim_vec_matrix[per_id, obj_id])
