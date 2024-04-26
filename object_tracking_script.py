import cv2
from detr_resnet_50 import object_tracking_detr_resnet

video_path = "../video_data/video_17s.mp4"
vid = cv2.VideoCapture(video_path) 

while(True):
	ret, frame = vid.read()
	object_tracking_detr_resnet()
	cv2.imshow('frame', frame) 
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 
cv2.destroyAllWindows() 
