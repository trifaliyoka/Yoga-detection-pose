import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
import tensorflow as tf
from collections import deque
import json

# ==============================================================================
# === 1. KONFIGURASI DAN SETUP ===
# ==============================================================================
print("🚀 1. Melakukan setup dan konfigurasi...")

# --- Path ke file-file penting ---
VIDEO_PATH = Path("hatta.mp4")
MODEL_PATH = Path("model_advanced_2/best_yoga_model_advanced.keras")
LABELS_PATH = Path("labels.csv") 
TEMPLATE_PATH = Path("yoga_angle_templates.json")

# --- Parameter Model & Aplikasi ---
SEQUENCE_LENGTH = 291
NUM_FEATURES = 166
CONFIDENCE_THRESHOLD = 0.75
FEEDBACK_DELAY_FRAMES = 30

# --- Inisialisasi MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==============================================================================
# === 2. MEMUAT SEMUA ASET (MODEL, LABEL, TEMPLATE) ===
# ==============================================================================
print("🧠 2. Memuat semua aset (Model, Label, Template Sudut)...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("   - Model berhasil dimuat.")
    labels_df = pd.read_csv(LABELS_PATH)
    index_to_class = {row["label_encoded"]: row["label"] for _, row in labels_df.iterrows()}
    print("   - Label berhasil dimuat.")
    with open(TEMPLATE_PATH, 'r') as f:
        angle_templates = json.load(f)
    print("   - Template sudut berhasil dimuat.")
except Exception as e:
    print(f"❌ KESALAHAN: Gagal memuat file penting. Error: {e}"); exit()

# ==============================================================================
# === 3. FUNGSI-FUNGSI PEMROSESAN LENGKAP ===
# ==============================================================================
print("🛠️  3. Menyiapkan fungsi-fungsi pemrosesan...")

def angle_between(p1, p2, p3):
    a=np.array(p1)-np.array(p2); b=np.array(p3)-np.array(p2)
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

def get_correction_feedback(current_angles, ideal_template):
    feedback = []
    for angle_name, user_angle in current_angles.items():
        if angle_name in ideal_template:
            bounds = ideal_template[angle_name]
            lower_bound, upper_bound = bounds['lower_bound_deg'], bounds['upper_bound_deg']
            if user_angle < lower_bound:
                feedback.append(f"Luruskan {angle_name.replace('_', ' ').title()}")
            elif user_angle > upper_bound:
                feedback.append(f"Tekuk {angle_name.replace('_', ' ').title()}")
    return feedback

# --- FUNGSI YANG HILANG SEBELUMNYA, SEKARANG DITAMBAHKAN KEMBALI ---
def extract_additional_features(frame_coords):
    """Ekstrak fitur geometris (jarak dan sudut) dari koordinat."""
    lm = mp_holistic.PoseLandmark
    pose = frame_coords[:33] # Ambil 33 landmark pose
    features = []
    
    # Kalkulasi Jarak
    points_to_process = [(lm.LEFT_SHOULDER, lm.LEFT_ELBOW), (lm.LEFT_ELBOW, lm.LEFT_WRIST), (lm.RIGHT_SHOULDER, lm.RIGHT_ELBOW), (lm.RIGHT_ELBOW, lm.RIGHT_WRIST), (lm.LEFT_HIP, lm.LEFT_KNEE), (lm.LEFT_KNEE, lm.LEFT_ANKLE), (lm.RIGHT_HIP, lm.RIGHT_KNEE), (lm.RIGHT_KNEE, lm.RIGHT_ANKLE)]
    for p1, p2 in points_to_process:
        features.append(np.linalg.norm(pose[p1.value] - pose[p2.value]))
        
    # Kalkulasi Sudut (dalam radian, karena model dilatih dengan ini)
    angles_to_process = [(lm.LEFT_SHOULDER, lm.LEFT_ELBOW, lm.LEFT_WRIST), (lm.RIGHT_SHOULDER, lm.RIGHT_ELBOW, lm.RIGHT_WRIST), (lm.LEFT_ELBOW, lm.LEFT_SHOULDER, lm.RIGHT_SHOULDER), (lm.RIGHT_ELBOW, lm.RIGHT_SHOULDER, lm.LEFT_SHOULDER), (lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE), (lm.RIGHT_HIP, lm.RIGHT_KNEE, lm.RIGHT_ANKLE), (lm.LEFT_KNEE, lm.LEFT_HIP, lm.RIGHT_HIP), (lm.RIGHT_KNEE, lm.RIGHT_HIP, lm.LEFT_HIP)]
    for p1, p2, p3 in angles_to_process:
        a, b = pose[p1.value] - pose[p2.value], pose[p3.value] - pose[p2.value]
        features.append(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6), -1.0, 1.0)))
        
    return np.array(features, dtype=np.float32)
# --- AKHIR DARI FUNGSI YANG HILANG ---

def process_frame_for_model(results, w, h):
    # Fungsi ini sama seperti sebelumnya, untuk membuat vektor fitur 166
    raw_kpts = [np.array([[lm.x,lm.y] for lm in res.landmark]) if res else np.zeros((n_lms,2)) for res,n_lms in [(results.pose_landmarks,33),(results.left_hand_landmarks,21),(results.right_hand_landmarks,21)]]
    s_l = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]; s_r = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    hu = abs((s_r.x - s_l.x) * w) / 2 if s_l and s_r else 0
    def norm_pose(lms,w,h,hu):
        if hu==0: return np.zeros((33,2))
        nx,ny=lms[0]*[w,h]; min_x,max_x=nx-hu*3,nx+hu*3; min_y,max_y=ny-hu,ny+hu*5
        dx,dy=max_x-min_x,max_y-min_y;
        if dx==0 or dy==0: return np.zeros((33,2))
        return np.round((lms*[w,h]-[min_x,min_y])/[dx,dy],12)
    def norm_hand(lms,w,h,hu):
        if np.sum(lms)==0: return np.zeros((21,2))
        xm,ym=np.mean(lms*[w,h],axis=0); dist=hu/1.5; min_x,max_x=xm-dist,xm+dist; min_y,max_y=ym-dist,ym+dist
        dx,dy=max_x-min_x,max_y-min_y
        if dx==0 or dy==0: return np.zeros((21,2))
        return np.round((lms*[w,h]-[min_x,min_y])/[dx,dy],12)
    norm_p = norm_pose(raw_kpts[0],w,h,hu); norm_lh = norm_hand(raw_kpts[1],w,h,hu); norm_rh = norm_hand(raw_kpts[2],w,h,hu)
    norm_vec = np.concatenate([norm_p,norm_lh,norm_rh]).flatten()
    add_fts = extract_additional_features(norm_vec.reshape(75, 2))
    return np.concatenate([norm_vec, add_fts])

# ==============================================================================
# === 4. PROSES INFERENSI REAL-TIME DENGAN KLASIFIKASI + KOREKSI ===
# ==============================================================================
print(f"\n🎥 4. Memulai inferensi real-time untuk '{VIDEO_PATH.name}'...")
# ... Sisa skrip sama persis seperti sebelumnya dan tidak perlu diubah ...
sequence_data = deque(maxlen=SEQUENCE_LENGTH)
prediction_text = "Mulai Pose Anda..."
feedback_list = []
feedback_timer = 0
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened(): print(f"❌ KESALAHAN: Tidak bisa membuka file video '{VIDEO_PATH.name}'."); exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True; image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        if results.pose_landmarks:
            h, w, _ = image.shape
            try:
                feature_vector = process_frame_for_model(results, w, h)
                sequence_data.append(feature_vector)
                
                if len(sequence_data) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence_data), axis=0)
                    prediction = model.predict(input_data, verbose=0)[0]
                    confidence = np.max(prediction)
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        predicted_index = np.argmax(prediction)
                        predicted_pose_name = index_to_class.get(predicted_index, "Tidak Dikenali")
                        prediction_text = f"{predicted_pose_name.replace('_', ' ').title()} ({confidence*100:.1f}%)"

                        ideal_template = angle_templates.get(predicted_pose_name)
                        if ideal_template and feedback_timer == 0:
                            pose_landmarks_for_angles = [[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark]
                            current_angles = calculate_all_angles(pose_landmarks_for_angles)
                            feedback_list = get_correction_feedback(current_angles, ideal_template)
                            feedback_timer = FEEDBACK_DELAY_FRAMES
                    else:
                        feedback_list = []
                else:
                    prediction_text = "Mengumpulkan Data..."
                    feedback_list = []

            except Exception as e:
                print(f"Error pada frame: {e}") # Cetak error spesifik untuk debug
                prediction_text = "Error Proses"
                feedback_list = []
        else:
            prediction_text = "Pose Tidak Terdeteksi"
            feedback_list = []

        cv2.putText(image, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        y_pos = 70
        for feedback in feedback_list:
            cv2.putText(image, feedback, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
            y_pos += 30
        
        if feedback_timer > 0:
            feedback_timer -= 1
            
        cv2.imshow('Pelatih Yoga Virtual', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Sesi Selesai.")