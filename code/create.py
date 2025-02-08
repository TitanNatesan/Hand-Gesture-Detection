import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Create directories to save gesture and image data
gesture_data_dir = "gesture_data"
image_data_dir = "gesture_images"
os.makedirs(gesture_data_dir, exist_ok=True)
os.makedirs(image_data_dir, exist_ok=True)

# Function to save gesture data to CSV
def save_gesture_data(gesture_name, landmarks):
    file_path = os.path.join(gesture_data_dir, f"{gesture_name}.csv")
    file_exists = os.path.exists(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            header = [f"landmark_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
            writer.writerow(header)
        writer.writerow(landmarks)

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'q' to exit the program.")

gesture_name = input("Enter the name of the gesture: ").strip()

# Create a folder for the gesture images
gesture_image_folder = os.path.join(image_data_dir, gesture_name)
os.makedirs(gesture_image_folder, exist_ok=True)

target_records = 500
saved_count = 0

# Check if any existing records already exist for this gesture
csv_file_path = os.path.join(gesture_data_dir, f"{gesture_name}.csv")
if os.path.exists(csv_file_path):
    with open(csv_file_path, "r") as file:
        saved_count = sum(1 for _ in file) - 1  # subtract header row

print(f"Starting collection for gesture: {gesture_name}")
print(f"Records already saved: {saved_count}/{target_records}")

while saved_count < target_records:
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            save_gesture_data(gesture_name, landmarks)

            # Save the associated image data to the gesture specific folder
            image_filename = f"{gesture_name}_{saved_count+1}.png"
            image_path = os.path.join(gesture_image_folder, image_filename)
            cv2.imwrite(image_path, frame)

            saved_count += 1
            print(f"Saved {saved_count}/{target_records} records for gesture: {gesture_name}")

    cv2.imshow("Hand Gesture Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

if saved_count >= target_records:
    print(f"Collected all {target_records} records for gesture: {gesture_name}.")
else:
    print(f"Collection incomplete: {saved_count}/{target_records} records saved.")

camera.release()
cv2.destroyAllWindows()
