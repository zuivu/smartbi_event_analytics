# import the opencv library 
import cv2

video_path = "../video_17s.mp4"
vid = cv2.VideoCapture(video_path) 

while(True):
	ret, frame = vid.read()
	

	cv2.imshow('frame', frame) 
	
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

vid.release() 
cv2.destroyAllWindows() 
