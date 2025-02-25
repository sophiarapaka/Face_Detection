import cv2

# Load Haar cascade classifier for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(frame):
    """
    Detects faces in a video frame and draws bounding boxes around them.

    Parameters:
    - frame: The video frame (image in BGR format).

    Returns:
    - faces: A list of bounding box coordinates for detected faces.
    """
    # Convert frame to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(40, 40)
    )

    # Draw bounding boxes around detected faces (if any)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Green box

    return faces

# Open webcam (0 for default camera)
video_capture = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not video_capture.isOpened():
    print("❌ Error: Could not access the webcam.")
else:
    print("✅ Webcam accessed successfully!")

# Start the real-time video capture loop
while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    if not result:
        print("❌ Error: Failed to capture frame")
        break  # Terminate the loop if the frame is not read successfully

    # Detect faces and draw bounding boxes
    detect_bounding_box(video_frame)

    # Display the processed frame in a window
    cv2.imshow("My Face Detection Project", video_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
print("✅ Webcam released. Program closed.")
