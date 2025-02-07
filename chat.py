import cv2
from deepface import DeepFace
import tensorflow as tf
import random
import time

# Set TensorFlow logging level
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Function to detect emotion using DeepFace
def detect_emotion():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)

        dominant_emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return dominant_emotion


# Sample empathetic responses based on emotion
def get_empathetic_response(emotion, user_input):
    responses = {
        'happy': [
            "I'm glad you're feeling happy! 😊 How can I assist you today?",
            "It's great to see you happy! What can I do for you?",
        ],
        'sad': [
            "I'm really sorry you're feeling down. 😢 Would you like to talk about it?",
            "I'm here for you. It's okay to feel sad sometimes. How can I help?",
        ],
        'angry': [
            "It seems like you're upset. 😠 Let's talk it through. What's on your mind?",
            "I can sense some frustration. If you want, we can work through it together.",
        ],
        'surprise': [
            "Wow, you look surprised! 😯 What happened?",
            "Something unexpected, I see! How can I help with that?",
        ],
        'neutral': [
            "I'm here to assist you. How can I help?",
            "Let me know what I can do for you.",
        ],
    }

    # Default response if no match
    default_response = "How can I help you today?"

    # Check if emotion is in the dictionary, otherwise use default response
    emotion_responses = responses.get(emotion.lower(), [default_response])

    # Respond with an empathetic reply
    return random.choice(emotion_responses)


# Function to simulate a chatbot conversation
def chatbot():
    while True:
        # Detect emotion from the user's face
        emotion = detect_emotion()

        # Prompt for text input
        user_input = input("You: ")

        # Get an empathetic response based on emotion and text
        response = get_empathetic_response(emotion, user_input)
        print(f"Chatbot: {response}")

        # Exit condition for the chatbot
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Adding a short pause for realistic interaction
        time.sleep(1)


# Run the chatbot
chatbot()
