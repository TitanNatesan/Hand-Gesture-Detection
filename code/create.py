import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import time
from collections import deque
import random
import math
from tqdm import tqdm


class HandGestureCollector:
    def __init__(self):
        # Initialize directories
        self.gesture_data_dir = "gesture_data"
        self.image_data_dir = "gesture_images"
        os.makedirs(self.gesture_data_dir, exist_ok=True)
        os.makedirs(self.image_data_dir, exist_ok=True)

        # MediaPipe setup with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on single hand for better quality
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,  # Using complex model for better tracking
        )

        # Collection settings
        self.target_records = 500
        self.preview_frames = 30  # Number of frames to preview before collecting
        self.quality_threshold = 0.7  # Minimum quality for sample acceptance
        self.stability_window = 10  # Frames to check for stable hand position

        # Buffers for data processing
        self.landmark_buffer = deque(maxlen=self.stability_window)
        self.quality_buffer = deque(maxlen=self.stability_window)
        self.augmentation_settings = {
            "rotation": (-15, 15),  # degrees
            "scale": (0.95, 1.05),  # scaling factor
            "translation": (-0.05, 0.05),  # as percentage of frame size
        }

    def normalize_landmarks(self, landmarks, frame_size):
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
            z = landmarks[
                i + 2
            ]  # Keep z-value as is, as it's already normalized by MediaPipe
            normalized.extend([x, y, z])

        return normalized

    def augment_landmarks(self, landmarks):
        """Apply random augmentations to landmarks to increase dataset diversity"""
        # Only augment after we have enough data to be representative
        augmented = landmarks.copy()

        # Apply random rotation
        rotation_angle = random.uniform(*self.augmentation_settings["rotation"]) * (
            math.pi / 180
        )
        scale_factor = random.uniform(*self.augmentation_settings["scale"])
        tx = random.uniform(*self.augmentation_settings["translation"])
        ty = random.uniform(*self.augmentation_settings["translation"])

        for i in range(0, len(augmented), 3):
            x, y = augmented[i], augmented[i + 1]

            # Rotate
            x_rot = x * math.cos(rotation_angle) - y * math.sin(rotation_angle)
            y_rot = x * math.sin(rotation_angle) + y * math.cos(rotation_angle)

            # Scale
            x_scaled = x_rot * scale_factor
            y_scaled = y_rot * scale_factor

            # Translate
            augmented[i] = x_scaled + tx
            augmented[i + 1] = y_scaled + ty

        return augmented

    def calculate_hand_quality(self, hand_landmarks, frame_size):
        """Calculate a quality score for the hand detection"""
        quality = 0.0

        # Check if all landmarks are within the frame boundaries
        all_in_frame = True
        landmark_visibility = []

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * frame_size[0]), int(lm.y * frame_size[1])
            if not (0 <= x < frame_size[0] and 0 <= y < frame_size[1]):
                all_in_frame = False

            # Estimate visibility based on z-coordinate (closer to camera = better)
            # MediaPipe z is negative as it goes away from camera
            visibility = 1.0 - min(1.0, max(0.0, lm.z + 0.2))
            landmark_visibility.append(visibility)

        # Base quality on visibility and if hand is in frame
        quality = sum(landmark_visibility) / len(landmark_visibility)
        if not all_in_frame:
            quality *= 0.5

        return quality

    def calculate_stability(self, landmarks):
        """Calculate how stable the hand position is over recent frames"""
        if len(self.landmark_buffer) < self.stability_window:
            return 0.0

        # Calculate average movement between consecutive frames
        movement = 0.0
        for i in range(1, len(self.landmark_buffer)):
            prev_landmarks = self.landmark_buffer[i - 1]
            curr_landmarks = self.landmark_buffer[i]

            # Calculate Euclidean distance for each point
            point_movement = 0.0
            for j in range(0, len(landmarks), 3):
                dx = curr_landmarks[j] - prev_landmarks[j]
                dy = curr_landmarks[j + 1] - prev_landmarks[j + 1]
                dz = curr_landmarks[j + 2] - prev_landmarks[j + 2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                point_movement += dist

            movement += point_movement / (len(landmarks) / 3)

        avg_movement = movement / (len(self.landmark_buffer) - 1)
        stability = max(
            0.0, 1.0 - (avg_movement * 10)
        )  # Scale for better visualization
        return stability

    def save_gesture_data(
        self, gesture_name, landmarks, normalized_landmarks, frame=None
    ):
        """Save the gesture data to CSV and optionally save the frame"""
        # Save to CSV
        file_path = os.path.join(self.gesture_data_dir, f"{gesture_name}.csv")
        file_exists = os.path.exists(file_path)

        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                header = [
                    f"landmark_{i}_{axis}"
                    for i in range(21)
                    for axis in ["x", "y", "z"]
                ]
                writer.writerow(header)
            writer.writerow(normalized_landmarks)

        # Save the image if provided
        if frame is not None:
            gesture_image_folder = os.path.join(self.image_data_dir, gesture_name)
            os.makedirs(gesture_image_folder, exist_ok=True)

            # Count existing files to determine the next index
            existing_files = len(os.listdir(gesture_image_folder))
            image_path = os.path.join(
                gesture_image_folder, f"{gesture_name}_{existing_files + 1}.jpg"
            )
            cv2.imwrite(image_path, frame)

    def count_existing_records(self, gesture_name):
        """Count how many records already exist for this gesture"""
        csv_file_path = os.path.join(self.gesture_data_dir, f"{gesture_name}.csv")
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "r") as file:
                return sum(1 for _ in file) - 1  # Subtract header row
        return 0

    def display_feedback(self, frame, gesture_name, saved_count, quality, stability):
        """Display collection progress and feedback on the frame"""
        # Draw progress bar
        progress = saved_count / self.target_records
        bar_width = 200
        bar_height = 20
        filled_width = int(bar_width * progress)

        cv2.rectangle(
            frame, (10, 70), (10 + bar_width, 70 + bar_height), (100, 100, 100), -1
        )
        cv2.rectangle(
            frame, (10, 70), (10 + filled_width, 70 + bar_height), (0, 255, 0), -1
        )

        # Quality and stability indicators
        quality_color = (0, int(255 * quality), int(255 * (1 - quality)))
        stability_color = (0, int(255 * stability), int(255 * (1 - stability)))

        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Progress: {saved_count}/{self.target_records}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Quality: {quality:.2f}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            quality_color,
            2,
        )
        cv2.putText(
            frame,
            f"Stability: {stability:.2f}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            stability_color,
            2,
        )

        if quality < self.quality_threshold:
            cv2.putText(
                frame,
                "Position hand clearly in frame",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return frame

    def collect_gestures(self):
        """Main method to collect hand gesture data"""
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open the camera.")
            return

        # Try to set higher resolution for better tracking
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\n=== Hand Gesture Data Collection ===")
        print("This tool will help you collect training data for hand gestures.")
        print("Position your hand clearly in the frame and hold each gesture steady.")
        print("\nPress 'q' at any time to exit the program.")

        while True:
            gesture_name = input(
                "\nEnter the name of the gesture to collect (or 'exit' to quit): "
            ).strip()
            if gesture_name.lower() == "exit":
                break

            saved_count = self.count_existing_records(gesture_name)
            print(f"\nStarting collection for gesture: {gesture_name}")
            print(f"Records already saved: {saved_count}/{self.target_records}")

            if saved_count >= self.target_records:
                print(f"Already collected enough samples for '{gesture_name}'.")
                continue_collection = input(
                    "Do you want to collect more? (y/n): "
                ).lower()
                if continue_collection != "y":
                    continue

            # Reset buffers
            self.landmark_buffer.clear()
            self.quality_buffer.clear()

            # Preview phase to get the user ready
            print("\nPreparing to collect. Position your hand and hold the gesture...")
            for _ in range(self.preview_frames):
                ret, frame = camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)  # Mirror effect
                cv2.putText(
                    frame,
                    f"Get ready to show: {gesture_name}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Hand Gesture Collection", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break

            # Collection phase with progress bar
            print(f"\nCollecting data for '{gesture_name}'...")
            collection_progress = tqdm(
                total=self.target_records - saved_count,
                desc=f"Collecting '{gesture_name}'",
            )

            last_save_time = time.time()
            min_save_interval = 0.1  # Maximum 10 frames per second

            while saved_count < self.target_records:
                ret, frame = camera.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                frame = cv2.flip(frame, 1)  # Mirror effect
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                quality = 0.0
                stability = 0.0
                frame_size = (frame.shape[1], frame.shape[0])

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style(),
                        )

                        # Extract landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        # Calculate quality and normalize landmarks
                        quality = self.calculate_hand_quality(
                            hand_landmarks, frame_size
                        )
                        normalized_landmarks = self.normalize_landmarks(
                            landmarks, frame_size
                        )

                        # Update buffers for stability calculation
                        self.landmark_buffer.append(normalized_landmarks)
                        self.quality_buffer.append(quality)

                        # Check if we have enough data for stability calculation
                        stability = (
                            self.calculate_stability(normalized_landmarks)
                            if len(self.landmark_buffer) >= self.stability_window
                            else 0.0
                        )

                        # Save data if quality is good enough and hand is stable
                        current_time = time.time()
                        time_elapsed = current_time - last_save_time

                        if (
                            time_elapsed >= min_save_interval
                            and quality >= self.quality_threshold
                            and stability >= 0.7
                        ):

                            # Create and save original sample
                            self.save_gesture_data(
                                gesture_name, landmarks, normalized_landmarks, frame
                            )
                            saved_count += 1
                            collection_progress.update(1)

                            # Create augmented samples (2 per real sample)
                            for _ in range(2):
                                augmented_landmarks = self.augment_landmarks(
                                    normalized_landmarks
                                )
                                self.save_gesture_data(
                                    gesture_name, landmarks, augmented_landmarks
                                )
                                saved_count += 1
                                collection_progress.update(1)

                            last_save_time = current_time

                # Display feedback
                frame = self.display_feedback(
                    frame, gesture_name, saved_count, quality, stability
                )
                cv2.imshow("Hand Gesture Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nExiting data collection...")
                    break

                if saved_count >= self.target_records:
                    print(f"\nCompleted collection for '{gesture_name}'!")
                    break

            collection_progress.close()

        camera.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nData collection completed. You can now train your model.")


def main():
    collector = HandGestureCollector()
    collector.collect_gestures()


if __name__ == "__main__":
    main()
