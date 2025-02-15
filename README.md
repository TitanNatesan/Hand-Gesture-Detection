# Hand Gesture Detection 🤚

Hand Gesture Detection using PyTorch and MediaPipe with Self-Generated Data 💻

## Table of Contents 📋
- [Hand Gesture Detection 🤚](#hand-gesture-detection-)
  - [Table of Contents 📋](#table-of-contents-)
  - [Description 📝](#description-)
  - [Features ✨](#features-)
  - [Installation 🛠️](#installation-️)
    - [Prerequisites](#prerequisites)
    - [Clone the Repository](#clone-the-repository)
    - [Install Dependencies](#install-dependencies)
  - [Usage 🏃](#usage-)
    - [Data Collection](#data-collection)
    - [Model Training](#model-training)
    - [Real-Time Recognition](#real-time-recognition)

## Description 📝
This project implements a hand gesture detection system using PyTorch and MediaPipe. It allows users to collect their own gesture data, train a custom neural network model, and perform real-time gesture recognition through a webcam.

## Features ✨
- **Data Collection:** Capture and save hand gesture landmarks using a webcam. 📸
- **Model Training:** Train a neural network model on the collected gesture data. 🧠
- **Real-Time Recognition:** Perform live gesture recognition with the trained model. 🔍
- **Scalability:** Easily add new gestures by collecting additional data. 📈

## Installation 🛠️

### Prerequisites
- Python 3.7 or higher 🐍
- pip

### Clone the Repository
```bash
git clone https://github.com/TitanNatesan/Hand-Gesture-Detection.git
cd Hand-Gesture-Detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage 🏃

### Data Collection
1. **Start Data Collection:**
      ```bash
        python create.py
      ```
2. **Follow the Prompts:**
                - Enter the name of the gesture you want to record.
                - Hold up the gesture in front of your webcam.
                - The script will save the gesture data automatically.

### Model Training
1. **Train the Model:**
      ```bash
        python train.py
      ```
2. **Training Details:**
                - The script will preprocess the data, train the neural network, and save the best model based on validation accuracy.

### Real-Time Recognition
1. **Run the Recognition Script:**
      ```bash
        python run.py
      ```
2. **Using the Application:**
                - The webcam will activate.
                - Perform a gesture in front of the camera.
                - The system will display the recognized gesture in real-time.
