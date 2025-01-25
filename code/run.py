import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

# Define the GestureNet model (must match the architecture in train.py)
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load the trained model and pre-saved scaler/label encoder classes from "Model" directory
model_dir = "Model"
model_path = os.path.join(model_dir, "hand_gesture_model.pth")
label_classes_path = os.path.join(model_dir, "label_encoder_classes.npy")
scaler_mean_path = os.path.join(model_dir, "scaler_mean.npy")
scaler_scale_path = os.path.join(model_dir, "scaler_scale.npy")

# Check if all required files exist
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()
if not os.path.exists(label_classes_path):
    print(f"Error: Label classes file '{label_classes_path}' not found.")
    exit()
if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_scale_path):
    print("Error: Scaler parameters not found.")
    exit()

# Load label encoder classes and scaler parameters
label_encoder_classes = np.load(label_classes_path)
scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
input_size = 63  # As per the training data
num_classes = len(label_encoder_classes)
model = GestureNet(input_size, num_classes).to(device)

# Load the saved model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from {model_path}")

# Standardize function using pre-saved scaler
def standardize_data(data):
    return (data - scaler_mean) / scaler_scale

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to exit the program.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If a hand is detected, display skeleton and make a prediction
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            # Extract landmarks (x, y, z for each point)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Standardize the data
            standardized_landmarks = standardize_data(np.array(landmarks).reshape(1, -1))

            # Convert to torch tensor
            input_tensor = torch.from_numpy(standardized_landmarks).float().to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = label_encoder_classes[predicted.item()]

            # Display the prediction on the frame
            cv2.putText(
                frame,
                f"Gesture: {predicted_label}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Display the frame
    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()