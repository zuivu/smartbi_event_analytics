def get_object_list(frames, names):
    """
    Extracts bounding boxes of persons and objects from a list of frames.
    Args:
        frames (list): A list of detection results for each frame. Each element of the list
            contains detection results such as bounding boxes, class IDs, and track IDs.
        names (dict): A dict of class names corresponding to the class IDs in the detection results.
    Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary contains bounding boxes of persons, where keys are track IDs
            and values are lists of location (x,y) # bounding box coordinates [x, y, width, height].
            - The second dictionary contains bounding boxes of objects other than persons, where
            keys are track IDs and values are lists of location (x,y) # bounding box coordinates [x, y, width, height].
    """
    person_list = {}
    object_list = {}
    for results in frames:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        conf = results[0].boxes.conf.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if names[class_id] == "person":
                add_to_dict(box.tolist(), track_id, person_list)
            else:
                add_to_dict(box.tolist(), track_id, object_list)

    return person_list, object_list


def add_to_dict(box, track_id, target_dict):
    # #### Modify by duy, only add x,y
    # x, y, w, h = box
    # location = (x,y)
    ####
    if track_id not in target_dict:
        target_dict[track_id] = [box]
    else:
        target_dict[track_id].append(box)
