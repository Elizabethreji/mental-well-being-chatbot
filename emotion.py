import cv2
from deepface import DeepFace
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def detect_emotion():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (DeepFace works with RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use DeepFace to analyze emotions
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        # Extract the dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']

        # Display the emotion on the video feed
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # If the user presses 'q', break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return dominant_emotion

# Call the function to detect emotion
emotion = detect_emotion()
print(f"Detected Emotion: {emotion}")
