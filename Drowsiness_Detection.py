import cv2
import mediapipe as mp
import numpy as np
import threading
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
import wave
import simpleaudio as sa

# Initialize Mediapipe modules
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Eye, mouth, and head landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 0, 17]
NOSE_TIP = 1
FOREHEAD = 10

# Load the sound file once at the beginning
try:
    with wave.open("alert.wav", "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())
        wave_obj = sa.WaveObject(audio_data, num_channels=wf.getnchannels(), bytes_per_sample=wf.getsampwidth(), sample_rate=wf.getframerate())
except FileNotFoundError:
    wave_obj = None
    print("Warning: 'alert.wav' not found. Audio alerts will not work.")
except Exception as e:
    wave_obj = None
    print(f"Error loading audio file: {e}. Please ensure 'alert.wav' is a valid WAV file.")

# Global variables for GUI and detection
drowsiness_counter = 0
yawn_counter = 0
head_nod_counter = 0
phone_counter = 0
seatbelt_counter = 0
drowsiness_score = 0
score_history = []
time_history = []
start_time = datetime.now()
is_running = False

# Function to play alert sound
def play_alert():
    if wave_obj:
        try:
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            print(f"Error playing sound: {e}")

# Function to write events to a CSV log file
def write_log(message):
    try:
        log_file = "activity_log.csv"
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Timestamp", "Event"])
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, message])
            print(f"Log written: {timestamp} - {message}")
    except Exception as e:
        print(f"Error writing to log file: {e}")

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(landmarks, eye_points, frame_w, frame_h):
    p1 = np.array([landmarks[eye_points[1]].x * frame_w, landmarks[eye_points[1]].y * frame_h])
    p2 = np.array([landmarks[eye_points[2]].x * frame_w, landmarks[eye_points[2]].y * frame_h])
    p3 = np.array([landmarks[eye_points[5]].x * frame_w, landmarks[eye_points[5]].y * frame_h])
    p4 = np.array([landmarks[eye_points[4]].x * frame_w, landmarks[eye_points[4]].y * frame_h])
    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p1 - p3)
    p0 = np.array([landmarks[eye_points[0]].x * frame_w, landmarks[eye_points[0]].y * frame_h])
    p3r = np.array([landmarks[eye_points[3]].x * frame_w, landmarks[eye_points[3]].y * frame_h])
    horizontal = np.linalg.norm(p0 - p3r)
    EAR = (vertical1 + vertical2) / (2.0 * horizontal)
    return EAR

# Mouth Aspect Ratio (MAR) function
def mouth_aspect_ratio(landmarks, mouth_points, frame_w, frame_h):
    p_top = np.array([landmarks[mouth_points[2]].x * frame_w, landmarks[mouth_points[2]].y * frame_h])
    p_bottom = np.array([landmarks[mouth_points[3]].x * frame_w, landmarks[mouth_points[3]].y * frame_h])
    vertical_dist = np.linalg.norm(p_top - p_bottom)
    p_left = np.array([landmarks[mouth_points[0]].x * frame_w, landmarks[mouth_points[0]].y * frame_h])
    p_right = np.array([landmarks[mouth_points[1]].x * frame_w, landmarks[mouth_points[1]].y * frame_h])
    horizontal_dist = np.linalg.norm(p_left - p_right)
    MAR = vertical_dist / horizontal_dist
    return MAR

# Head Pose Estimation function
def head_pose_estimation(landmarks, nose_tip_idx, forehead_idx):
    nose_tip_3d = np.array([landmarks[nose_tip_idx].x, landmarks[nose_tip_idx].y, landmarks[nose_tip_idx].z])
    forehead_3d = np.array([landmarks[forehead_idx].x, landmarks[forehead_idx].y, landmarks[forehead_idx].z])
    head_vector = nose_tip_3d - forehead_3d
    pitch_angle_rad = np.arctan2(head_vector[1], np.sqrt(head_vector[0]**2 + head_vector[2]**2))
    pitch_angle_deg = np.degrees(pitch_angle_rad)
    return pitch_angle_deg

# Main video processing loop
def start_detection():
    global drowsiness_counter, yawn_counter, head_nod_counter, phone_counter, seatbelt_counter, drowsiness_score, is_running
    
    if is_running:
        messagebox.showinfo("Status", "Detection is already running.")
        return

    is_running = True
    cap = cv2.VideoCapture(0)
    
    # Thresholds
    EAR_THRESHOLD = 0.25
    CLOSED_FRAMES = 90
    MAR_THRESHOLD = 0.70
    YAWN_FRAMES = 15
    PITCH_THRESHOLD = -15.0
    HEAD_NOD_FRAMES = 20
    PHONE_DETECT_FRAMES = 15
    SEATBELT_FRAMES = 10

    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Processing for hands and face
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        drowsiness_score = 0
        final_status = "Awake & Alert"
        
        # --- Seatbelt Detection ---
        seatbelt_roi = frame[int(h*0.3):int(h*0.6), int(w*0.3):int(w*0.7)]
        gray_roi = cv2.cvtColor(seatbelt_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for sharp lines (indicating a seatbelt)
        edges = cv2.Canny(gray_roi, 50, 150)
        line_count = np.sum(edges > 0)
        
        if line_count < 1500: # Threshold can be adjusted
            seatbelt_counter += 1
            if seatbelt_counter >= SEATBELT_FRAMES:
                final_status = "❌ Seatbelt Not Fastened!"
                if seatbelt_counter == SEATBELT_FRAMES:
                    write_log("Seatbelt Not Fastened")
                    threading.Thread(target=play_alert, daemon=True).start()
        else:
            seatbelt_counter = 0

        # --- Phone Detection ---
        phone_in_hand = False
        if hand_results.multi_hand_landmarks:
            phone_counter += 1
            if phone_counter >= PHONE_DETECT_FRAMES:
                phone_in_hand = True
                final_status = "📵 Phone Detected!"
                if phone_counter == PHONE_DETECT_FRAMES:
                    write_log("Phone Usage Detected")
                    threading.Thread(target=play_alert, daemon=True).start()
        else:
            phone_counter = 0

        # --- Drowsiness/Yawning/Head Nod Detection ---
        if not final_status.startswith(("❌", "📵")) and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_EAR = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
                right_EAR = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
                avg_EAR = (left_EAR + right_EAR) / 2.0
                if avg_EAR < EAR_THRESHOLD:
                    drowsiness_counter += 1
                    drowsiness_score += 40
                else:
                    drowsiness_counter = 0
                
                mouth_MAR = mouth_aspect_ratio(face_landmarks.landmark, MOUTH, w, h)
                if mouth_MAR > MAR_THRESHOLD:
                    yawn_counter += 1
                    drowsiness_score += 30
                else:
                    yawn_counter = 0
                
                pitch_angle = head_pose_estimation(face_landmarks.landmark, NOSE_TIP, FOREHEAD)
                if pitch_angle < PITCH_THRESHOLD:
                    head_nod_counter += 1
                    drowsiness_score += 30
                else:
                    head_nod_counter = 0
                
                if drowsiness_counter >= CLOSED_FRAMES:
                    final_status = "😴 Drowsiness Detected!"
                    if drowsiness_counter == CLOSED_FRAMES:
                        write_log("Drowsiness Detected (Eyes Closed)")
                        threading.Thread(target=play_alert, daemon=True).start()
                elif yawn_counter >= YAWN_FRAMES:
                    final_status = "🥱 Yawning Detected!"
                    if yawn_counter == YAWN_FRAMES:
                        write_log("Yawning Detected")
                        threading.Thread(target=play_alert, daemon=True).start()
                elif head_nod_counter >= HEAD_NOD_FRAMES:
                    final_status = "😴 Head Nod Detected!"
                    if head_nod_counter == HEAD_NOD_FRAMES:
                        write_log("Head Nod Detected")
                        threading.Thread(target=play_alert, daemon=True).start()
                
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )

        # Update GUI labels
        status_label.config(text=f"Status: {final_status}")
        score_label.config(text=f"Drowsiness Score: {drowsiness_score}")
        
        # Update graph data
        current_time = (datetime.now() - start_time).total_seconds()
        score_history.append(drowsiness_score)
        time_history.append(current_time)
        
        if len(score_history) > 50:
            score_history.pop(0)
            time_history.pop(0)
        
        update_graph()

        cv2.imshow("Driver Drowsiness and Activity Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    is_running = False

def stop_detection():
    global is_running
    if not is_running:
        messagebox.showinfo("Status", "Detection is already stopped.")
        return
    is_running = False

def update_graph():
    ax.clear()
    ax.plot(time_history, score_history, color='blue')
    ax.set_title('Real-time Drowsiness Score')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    canvas.draw()

# --- GUI Setup (Tkinter) ---
root = tk.Tk()
root.title("Driver Drowsiness Monitoring System")
root.geometry("800x600")

# Title Label
title_label = tk.Label(root, text="Driver Drowsiness Monitor", font=("Arial", 24))
title_label.pack(pady=10)

# Status and score labels
status_label = tk.Label(root, text="Status: Not Running", font=("Arial", 16))
status_label.pack(pady=5)
score_label = tk.Label(root, text="Drowsiness Score: 0", font=("Arial", 16))
score_label.pack(pady=5)

# Start/Stop buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
start_button = tk.Button(button_frame, text="Start Detection", command=lambda: threading.Thread(target=start_detection).start(), font=("Arial", 14), bg="green", fg="white")
start_button.pack(side=tk.LEFT, padx=10)
stop_button = tk.Button(button_frame, text="Stop Detection", command=stop_detection, font=("Arial", 14), bg="red", fg="white")
stop_button.pack(side=tk.RIGHT, padx=10)

# Matplotlib graph
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=20)

root.mainloop()