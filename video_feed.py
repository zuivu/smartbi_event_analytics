import os
import cv2
from dotenv import load_dotenv

load_dotenv()
USERNAME = os.environ.get("USERNAME")
PASSWORD = os.environ.get("PASSWORD")
IP_ADDRESS = os.environ.get("IP_ADDRESS")
high_res = True

def display_rtsp_stream(rtsp_url):
    # Create a video capture object with the RTSP URL
    cap = cv2.VideoCapture(rtsp_url)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    try:
        # Loop to continuously fetch frames from the RTSP stream
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv2.imshow('RTSP Stream', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_id = "stream1" if high_res else "stream2"
    rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/{stream_id}"
    display_rtsp_stream(rtsp_url)
