# pose_analysis.py

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import tensorflow as tf
import mediapipe as mp
from collections import deque

# --- Konstanta ---
MODEL_PATH = Path("model_advanced_2/best_yoga_model_advanced.keras")
LABELS_PATH = Path("labels.csv")
TEMPLATE_PATH = Path("yoga_angle_templates.json")
SEQUENCE_LENGTH = 291
NUM_FEATURES = 166
CONFIDENCE_THRESHOLD = 0.75

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic

# --- Load model dan label ---
model = tf.keras.models.load_model(MODEL_PATH)
labels_df = pd.read_csv(LABELS_PATH)
index_to_class = {row["label_encoded"]: row["label"] for _, row in labels_df.iterrows()}

with open(TEMPLATE_PATH, 'r') as f:
    angle_templates = json.load(f)

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), -1.0, 1.0)))

def calculate_all_angles(pose_landmarks):
    lm = mp_holistic.PoseLandmark
    angles = {}
    angles['left_elbow'] = angle_between(pose_landmarks[lm.LEFT_SHOULDER.value], pose_landmarks[lm.LEFT_ELBOW.value], pose_landmarks[lm.LEFT_WRIST.value])
    angles['right_elbow'] = angle_between(pose_landmarks[lm.RIGHT_SHOULDER.value], pose_landmarks[lm.RIGHT_ELBOW.value], pose_landmarks[lm.RIGHT_WRIST.value])
    angles['left_shoulder'] = angle_between(pose_landmarks[lm.LEFT_ELBOW.value], pose_landmarks[lm.LEFT_SHOULDER.value], pose_landmarks[lm.LEFT_HIP.value])
    angles['right_shoulder'] = angle_between(pose_landmarks[lm.RIGHT_ELBOW.value], pose_landmarks[lm.RIGHT_SHOULDER.value], pose_landmarks[lm.RIGHT_HIP.value])
    angles['left_hip'] = angle_between(pose_landmarks[lm.LEFT_SHOULDER.value], pose_landmarks[lm.LEFT_HIP.value], pose_landmarks[lm.LEFT_KNEE.value])
    angles['right_hip'] = angle_between(pose_landmarks[lm.RIGHT_SHOULDER.value], pose_landmarks[lm.RIGHT_HIP.value], pose_landmarks[lm.RIGHT_KNEE.value])
    angles['left_knee'] = angle_between(pose_landmarks[lm.LEFT_HIP.value], pose_landmarks[lm.LEFT_KNEE.value], pose_landmarks[lm.LEFT_ANKLE.value])
    angles['right_knee'] = angle_between(pose_landmarks[lm.RIGHT_HIP.value], pose_landmarks[lm.RIGHT_KNEE.value], pose_landmarks[lm.RIGHT_ANKLE.value])
    return angles

def get_feedback(current_angles, ideal_template):
    feedback = []
    for angle_name, user_angle in current_angles.items():
        if angle_name in ideal_template:
            bounds = ideal_template[angle_name]
            if user_angle < bounds["lower_bound_deg"]:
                feedback.append(f"Luruskan {angle_name.replace('_', ' ').title()}")
            elif user_angle > bounds["upper_bound_deg"]:
                feedback.append(f"Tekuk {angle_name.replace('_', ' ').title()}")
    return feedback

def extract_features(results, w, h):
    lm = mp_holistic.PoseLandmark
    def norm(lms):
        if not lms:
            return np.zeros((33, 2))
        s_l = results.pose_landmarks.landmark[lm.LEFT_SHOULDER]
        s_r = results.pose_landmarks.landmark[lm.RIGHT_SHOULDER]
        hu = abs((s_r.x - s_l.x) * w) / 2 if s_l and s_r else 0
        pose = [[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark]
        return np.array(pose), hu

    pose, _ = norm(results.pose_landmarks.landmark if results.pose_landmarks else [])
    return np.concatenate([pose.flatten(), np.zeros(NUM_FEATURES - len(pose.flatten()))])

def analyze_video(video_path, output_frame_path):
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)
    cap = cv2.VideoCapture(str(video_path))
    prediction_text = "Pose Tidak Terdeteksi"
    feedback_list = []
    first_frame_saved = False

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            if results.pose_landmarks:
                features = extract_features(results, w, h)
                sequence_data.append(features)

                if not first_frame_saved:
                    cv2.imwrite(output_frame_path, frame)
                    first_frame_saved = True

                if len(sequence_data) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence_data), axis=0)
                    prediction = model.predict(input_data, verbose=0)[0]
                    confidence = np.max(prediction)
                    if confidence > CONFIDENCE_THRESHOLD:
                        predicted_index = np.argmax(prediction)
                        label = index_to_class[predicted_index]
                        prediction_text = f"{label.title()} ({confidence*100:.1f}%)"
                        pose_landmarks_for_angles = [[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark]
                        feedback_list = get_feedback(calculate_all_angles(pose_landmarks_for_angles), angle_templates[label])
                    break
    cap.release()
    return prediction_text, feedback_list
