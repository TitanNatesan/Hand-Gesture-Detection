import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time

# Set path for the model files
model_dir = "models"
model_path = os.path.join(model_dir, "hand_gesture_model.h5")
tflite_model_path = os.path.join(model_dir, "hand_gesture_model.tflite")
scaler_path = os.path.join(model_dir, "scaler.joblib")
label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")

# Check if model files exist
if not os.path.exists(model_path) and not os.path.exists(tflite_model_path):
    print(f"Error: No model files found in {model_dir}")
    print("Please run train.py first to create the model.")
    exit()

# Load the scaler and label encoder
if os.path.exists(scaler_path) and os.path.exists(label_encoder_path):
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    print(f"Loaded preprocessing tools from {model_dir}")
    print(f"Available gestures: {', '.join(label_encoder.classes_)}")
else:
    print("Error: Preprocessing files not found. Please run train.py first.")
    exit()

# Determine which model to use
use_tflite = False  # Set to True to use the optimized TFLite model

if use_tflite:
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Loaded TFLite model from {tflite_model_path}")

    # Define prediction function for TFLite
    def predict(input_data):
        input_data = input_data.astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        return output

else:
    # Load Keras model
    try:
        model = load_model(model_path)
        print(f"Loaded Keras model from {model_path}")

        # Define prediction function for Keras model
        def predict(input_data):
            return model.predict(input_data, verbose=0)

    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Focus on single hand for better quality
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


# Function to normalize landmarks similar to what we did in training
def normalize_landmarks(landmarks, frame_size):
    """Normalize landmarks relative to hand size for better generalization"""
    # Extract all x and y coordinates
    x_coords = [landmarks[i] for i in range(0, len(landmarks), 3)]
    y_coords = [landmarks[i + 1] for i in range(0, len(landmarks), 3)]

    # Find bounding box of hand
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate center and size
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    bbox_size = max(max_x - min_x, max_y - min_y, 0.1)  # Avoid division by zero

    # Normalize landmarks to be centered and scaled
    normalized = []
    for i in range(0, len(landmarks), 3):
        x = (landmarks[i] - center_x) / bbox_size
        y = (landmarks[i + 1] - center_y) / bbox_size
        z = landmarks[i + 2]  # Keep z-value as is
        normalized.extend([x, y, z])

    return normalized


# Open the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Try to set higher resolution for better tracking
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to exit the program.")
print("Press 'm' to toggle between standard model and TFLite model.")

# Variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

# Variable to store prediction confidence
confidence = 0
prediction_label = ""

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the frame as not writeable
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Display which model is active
    model_text = "TFLite Model" if use_tflite else "Standard Model"
    cv2.putText(
        frame,
        model_text,
        (frame.shape[1] - 200, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Process hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Extract landmarks
            landmarks = []
            frame_size = (frame.shape[1], frame.shape[0])
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Normalize landmarks same way as in training
            normalized_landmarks = normalize_landmarks(landmarks, frame_size)

            # Apply scaler transformation
            scaled_landmarks = scaler.transform(
                np.array(normalized_landmarks).reshape(1, -1)
            )

            # Make prediction
            prediction = predict(scaled_landmarks)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class_idx]
            prediction_label = label_encoder.classes_[predicted_class_idx]

    # Display the prediction and confidence
    if prediction_label:
        # Color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence

        cv2.putText(
            frame,
            f"Gesture: {prediction_label}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # Display the frame
    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Exiting...")
        break
    elif key == ord("m"):
        # Toggle between standard and TFLite model
        use_tflite = not use_tflite
        print(f"Switched to {'TFLite' if use_tflite else 'Standard'} model")

        if use_tflite:
            # Reinitialize TFLite interpreter if needed
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Update prediction function
            def predict(input_data):
                input_data = input_data.astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                return interpreter.get_tensor(output_details[0]["index"])

        else:
            # Switch back to standard model
            def predict(input_data):
                return model.predict(input_data, verbose=0)


# Release resources
camera.release()
cv2.destroyAllWindows()
hands.close()
