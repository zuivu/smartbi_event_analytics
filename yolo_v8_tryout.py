import os
import json
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
import pickle

from get_object_list import get_object_list
from predict_trajectory_vector import predict_trajectory_vector
from get_attraction_matrix import get_attraction_matrix, get_similarity_vector_matrix, get_attraction_matrix_from_boxes

from dotenv import load_dotenv

load_dotenv()
USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")
IP_ADDRESS = os.environ.get("IP_ADDRESS")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/stream1" 
#video_path = "../our_data/surveillance_camera_3.mp4"
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
        
        # Visualization at each frame
        annotated_frame = results[0].plot(conf=False, labels=False, boxes=False)

        ## Get the boxes and track IDs
        if results[0].boxes.id is not None:
            # On every frame
            frames_list.append(results)

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            attractive_object_indices = []
            persons_filtered_idx = []
            objects_filtered_idx = []
            
            person_dict, object_dict = get_object_list(frames_list[-10:], names=results[0].names)
            if len(person_dict) >= 1 and len(object_dict) >= 1:
                # Get trajectories and location of persons of interest (appear at least a certain number of frames, defauly by 5) 
                persons_location, persons_trajectory, persons_pred_traj, persons_filtered_idx = predict_trajectory_vector(person_dict, threshold=3)
                objects_location, _, _, objects_filtered_idx = predict_trajectory_vector(object_dict)

                # Visualize predicted path, size (number of objects, number of points, 2)
                if len(persons_pred_traj) > 0:
                    predicted_path = np.hstack(persons_pred_traj.reshape(-1,2)).astype(np.int32).reshape((-1, 1, 2))
                    for per_id in range(len(persons_location)):
                        cv2.arrowedLine(annotated_frame,
                                        persons_location[per_id],
                                        persons_pred_traj[per_id],
                                        color=(255, 0, 0),
                                        tipLength=0.15,
                                        thickness=8)
                        # for object_id in range(len(objects_location)):
                        #     cv2.arrowedLine(annotated_frame,
                        #                     persons_location[per_id],
                        #                     objects_location[object_id],
                        #                     color=(120, 240, 120),
                        #                     tipLength=0.1,
                        #                     thickness=5)

                    if len(objects_location) != 0: # in case object does not pass the threshold
                        # Get cosine matrix
                        attraction_matrix = get_attraction_matrix(persons_location, objects_location)
                        cos_sim_path_matrix = get_similarity_vector_matrix(attraction_matrix, persons_trajectory)

                        # Get truth table of attention,
                        # if cos > 0.2 (0 mean trajectory is orthogonal with direction to object,
                        # the closer to 1, the more attraction it get)
                        sim_thres = 0.7
                        sim_path_result = (cos_sim_path_matrix >= sim_thres)
                        
                        # then if more than 3 out of 10 people in the period has path closed, then object is attractive
                        attractive_thres = 1
                        sim_path_count_per_object = np.sum(sim_path_result, axis=0)
                        threshold = attractive_thres * sim_path_count_per_object.shape[0]
                        # 60% of total rows
                        attractive_object_indices = np.array(objects_filtered_idx)[sim_path_count_per_object.astype(bool)]  # .astype(bool)

            # Draw box of attractive objects and trajectory of person
            for box, track_id in zip(boxes, track_ids):
                if track_id in persons_filtered_idx: # only draw tracking for humans
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 15:  # retain 15 tracks for 90 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 255), thickness=12)
                else:
                    if track_id in attractive_object_indices:
                        x,y,w,h = box
                        x = int(x - w/2)
                        y = int(y-h/2)
                        w = int(w)
                        h = int(h)
                        cv2.rectangle(annotated_frame, (x + w, y + h), (x, y), (0, 0, 255), 10)

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


# with open("results.pickle", "wb") as f:
#     pickle.dump(frames_list, f)