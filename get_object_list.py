def get_object_list(frames):
    person_list = {}
    object_list = {}
    for results in frames:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        object_classes = results[0].boxes.cls.str().cpu().tolist()

        for box, track_id, object_class in zip(boxes, track_ids, object_classes):
            if object_class == "Person":
                add_to_dict(box, track_id, person_list)
            else:
                add_to_dict(box, track_id, object_list)

    return person_list, object_list

def add_to_dict(box, track_id, target_dict):
    if track_id not in target_dict:
        target_dict[track_id] = [box]
    else:
        target_dict[track_id].append(box)