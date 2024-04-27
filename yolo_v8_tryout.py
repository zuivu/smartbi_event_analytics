import json
from collections import defaultdict

import cv2
import time
import math
import numpy as np
from ultralytics import YOLO

from get_object_list import get_object_list
from predict_trajectory_vector import predict_trajectory_vector
from get_attraction_matrix import get_attraction_matrix, get_similarity_vector_matrix


# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "../our_data/surveillance_camera_3.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Initialize for additional detection
frame_count = 0 
frames_list = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # On every frame
        frames_list.append(results)

        # Visualization at each frame
        annotated_frame = results[0].plot()



        ## Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        attractive_object_indices = []


        
        person_dict, object_dict = get_object_list(frames_list[-30:], names=results[0].names)
        if len(person_dict) >= 1 and len(object_dict) >= 1:
            # Get trajectories and location of persons of interest (appear at least a certain number of frames, defauly by 5) 
            persons_location, persons_trajectory, persons_pred_traj = predict_trajectory_vector(person_dict)
            objects_location, _, _ = predict_trajectory_vector(object_dict)

            # Visualize predicted path, size (number of objects, number of points, 2)
            if len(persons_pred_traj) > 0:
                predicted_path = np.hstack(persons_pred_traj.reshape(-1,2)).astype(np.int32).reshape((-1, 1, 2))
                for per_id in range(len(persons_location)):
                    cv2.arrowedLine(annotated_frame,
                                    persons_location[per_id],
                                    persons_pred_traj[per_id],
                                    color=(128, 128, 128),
                                    tipLength=0.15,
                                    thickness=8)

                
        # On every 30 frames
        if len(person_dict) >= 1 and len(object_dict) >= 1 and frame_count % 30 == 0:
            # Get persons and objects list
            # person_dict, object_dict = get_object_list(frames_list, names=results[0].names)
            
            # # Get trajectories and location of persons of interest (appear at least a certain number of frames, defauly by 5) 
            # persons_location, persons_trajectory, persons_pred_traj = predict_trajectory_vector(person_dict)
            # objects_location, objects_trajectory, _ = predict_trajectory_vector(object_dict)

            # # Visualize predicted path, size (number of objects, number of points, 2)
            # predicted_path = np.hstack(persons_pred_traj.reshape(-1,2)).astype(np.int32).reshape((-1, 1, 2))
            # for per_id in range(len(persons_location)):
            #     cv2.arrowedLine(annotated_frame,
            #                     persons_location[per_id],
            #                     persons_pred_traj[per_id],
            #                     color=(230, 230, 230),
            #                     tipLength=0.05,
            #                     thickness=10)  
            
            # Get cosine matrix
            attraction_matrix = get_attraction_matrix(persons_location, objects_location)
            cos_sim_path_matrix = get_similarity_vector_matrix(attraction_matrix, persons_trajectory)

            # Get truth table of attention,
            # if cos < 0.2 then, path is closed,
            sim_thres = 0.2
            attractive_thres = 0.6
            sim_path_result = np.logical_and(cos_sim_path_matrix >= -sim_thres, cos_sim_path_matrix <= sim_thres)
            # then if more than 60% of people in the period has path closed, then object is attractive
            sim_path_count_per_object = np.sum(sim_path_result, axis=0)
            threshold = attractive_thres * sim_path_count_per_object.shape[0]
            # 60% of total rows
            attractive_object_indices = np.where(sim_path_count_per_object > threshold)[0].tolist()

            # Reset count and frames collection 
            # frame_count = 0
            #frames_list = []
        


        #### TODO: need to recustomize bounding box to highlight attractive objects
        #attractive_object_indices






        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


with open("tracking_data.json", "wb") as f:
    str = json.dumps(track_history)
    f.write(str.encode())