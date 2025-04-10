import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import random
import glob
import joblib
from tqdm import tqdm
import time


class HandGestureTrainer:
    def __init__(self, data_dir="gesture_data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.batch_size = 64
        self.epochs = 100
        self.input_dim = 63  # 21 landmarks × 3 (x, y, z)
        self.model = None
        self.history = None
        self.gesture_names = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Set seeds for reproducibility
        self.set_seeds(42)

    def set_seeds(self, seed):
        """Set seeds for reproducibility"""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def load_data(self):
        """Load all gesture data from CSV files"""
        print("Loading gesture data...")
        all_data = []
        all_labels = []

        # Get all CSV files in the data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        for csv_file in tqdm(csv_files):
            gesture_name = os.path.basename(csv_file).replace(".csv", "")
            self.gesture_names.append(gesture_name)

            try:
                df = pd.read_csv(csv_file)
                data = df.values
                labels = [gesture_name] * len(data)

                all_data.append(data)
                all_labels.extend(labels)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        # Combine all data
        X = np.vstack(all_data)
        y = np.array(all_labels)

        # Encode labels
        self.gesture_names = sorted(list(set(self.gesture_names)))
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Loaded {len(X)} samples across {len(self.gesture_names)} gestures")
        print(f"Gestures: {', '.join(self.gesture_names)}")

        return X, y_encoded

    def preprocess_data(self, X, y):
        """Preprocess data with normalization and train/test split"""
        print("Preprocessing data...")

        # Scale features for better training performance
        X_scaled = self.scaler.fit_transform(X)

        # Convert labels to one-hot encoding
        y_onehot = to_categorical(y)

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        return X_train, X_val, y_train, y_val

    def augment_data(self, X_train, y_train, augment_factor=0.5):
        """Apply additional augmentation to the training data"""
        print("Applying additional data augmentation...")

        num_samples = int(len(X_train) * augment_factor)
        indices = np.random.choice(len(X_train), num_samples, replace=False)

        X_aug = X_train[indices].copy()
        y_aug = y_train[indices].copy()

        # Apply random noise and perturbations
        for i in range(len(X_aug)):
            # Add small random noise
            noise = np.random.normal(0, 0.01, X_aug[i].shape)
            X_aug[i] += noise

            # Random scaling (simulating different hand sizes)
            scale_factor = np.random.uniform(0.95, 1.05)
            # Apply scaling to each landmark's x, y coordinates
            for j in range(0, X_aug.shape[1], 3):
                X_aug[i, j] *= scale_factor  # x coordinate
                X_aug[i, j + 1] *= scale_factor  # y coordinate

            # Z-axis perturbations
            for j in range(2, X_aug.shape[1], 3):
                X_aug[i, j] += np.random.normal(0, 0.02)

        # Combine original and augmented data
        X_combined = np.vstack([X_train, X_aug])
        y_combined = np.vstack([y_train, y_aug])

        print(f"Augmented training set size: {len(X_combined)}")
        return X_combined, y_combined

    def build_model(self, num_classes, model_type="nn"):
        """Build the model architecture based on the specified type"""
        print(f"Building {model_type} model...")

        if model_type == "nn":
            # Neural network optimized for landmarks
            model = Sequential(
                [
                    # Input noise layer for better generalization
                    GaussianNoise(0.01, input_shape=(self.input_dim,)),
                    # First hidden layer
                    Dense(128, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.3),
                    # Second hidden layer
                    Dense(64, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.2),
                    # Third hidden layer with smaller neurons for efficiency
                    Dense(32, activation="relu"),
                    BatchNormalization(),
                    Dropout(0.1),
                    # Output layer
                    Dense(num_classes, activation="softmax"),
                ]
            )

            # Compile with Adam optimizer and learning rate scheduling
            optimizer = Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Print model summary
        model.summary()
        self.model = model
        return model

    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the model with callbacks for better performance"""
        print("Training model...")

        # Define callbacks for training
        callbacks = [
            # Stop training when validation doesn't improve
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            # Save best model during training
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            # Reduce learning rate when validation plateaus
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ]

        # Train the model
        start_time = time.time()
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        self.history = history
        return history

    def evaluate_model(self, X_val, y_val):
        """Evaluate the model and print classification metrics"""
        print("\nEvaluating model performance...")

        # Get overall accuracy
        loss, accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation loss: {loss:.4f}")

        # Get predicted classes
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_true_classes, y_pred_classes, target_names=self.gesture_names
            )
        )

        # Calculate and plot confusion matrix
        self.plot_confusion_matrix(y_true_classes, y_pred_classes)

        # Plot training history
        self.plot_training_history()

        return accuracy

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot the confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)

        # Normalize the confusion matrix
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.gesture_names,
            yticklabels=self.gesture_names,
        )

        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "confusion_matrix.png"))
        plt.close()

    def plot_training_history(self):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="lower right")

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_history.png"))
        plt.close()

    def optimize_for_inference(self):
        """Convert model to TFLite for faster inference"""
        print("\nOptimizing model for inference...")

        # Save the standard model
        standard_model_path = os.path.join(self.model_dir, "hand_gesture_model.h5")
        self.model.save(standard_model_path)
        print(f"Standard model saved to {standard_model_path}")

        # Save the scaler and label encoder
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        joblib.dump(
            self.label_encoder, os.path.join(self.model_dir, "label_encoder.joblib")
        )

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Convert the model
        tflite_model = converter.convert()

        # Save the TFLite model
        tflite_model_path = os.path.join(self.model_dir, "hand_gesture_model.tflite")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"Optimized TFLite model saved to {tflite_model_path}")

        # Calculate model size
        standard_size = os.path.getsize(standard_model_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)

        print(f"Standard model size: {standard_size:.2f} MB")
        print(f"Optimized TFLite model size: {tflite_size:.2f} MB")
        print(f"Size reduction: {(1 - tflite_size/standard_size) * 100:.1f}%")

    def test_inference_speed(self, X_test, num_tests=100):
        """Test and report inference speed"""
        print("\nTesting inference speed...")

        # Warm up
        for _ in range(10):
            self.model.predict(X_test[0:1])

        # Test batch inference speed
        start_time = time.time()
        for _ in range(num_tests):
            self.model.predict(X_test[0:1])
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / num_tests * 1000  # ms
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Inference speed: {1000/avg_inference_time:.1f} FPS")

        # Try to load and test the TFLite model
        try:
            # Load TFLite model
            tflite_path = os.path.join(self.model_dir, "hand_gesture_model.tflite")
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Test TFLite inference speed
            start_time = time.time()
            for _ in range(num_tests):
                input_data = X_test[0:1].astype(np.float32)
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]["index"])
            end_time = time.time()

            avg_tflite_time = (end_time - start_time) / num_tests * 1000  # ms
            print(f"TFLite average inference time: {avg_tflite_time:.2f} ms")
            print(f"TFLite inference speed: {1000/avg_tflite_time:.1f} FPS")
            print(f"Speed improvement: {avg_inference_time/avg_tflite_time:.1f}x")

        except Exception as e:
            print(f"TFLite testing failed: {e}")

    def train(self):
        """Full training pipeline"""
        # Load data
        X, y = self.load_data()

        # Preprocess data
        X_train, X_val, y_train, y_val = self.preprocess_data(X, y)

        # Augment training data
        X_train, y_train = self.augment_data(X_train, y_train)

        # Build model
        self.build_model(num_classes=len(self.gesture_names))

        # Train model
        self.train_model(X_train, X_val, y_train, y_val)

        # Evaluate model
        self.evaluate_model(X_val, y_val)

        # Optimize model for inference
        self.optimize_for_inference()

        # Test inference speed
        self.test_inference_speed(X_val)

        print("\nTraining and optimization completed successfully!")
        print(f"Model files saved in {self.model_dir}")


def main():
    trainer = HandGestureTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
