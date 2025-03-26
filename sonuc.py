import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = joblib.load('tennis_action_model.pkl')

path_movie = 'C:\\Users\\user\\Desktop\\tenis\\tenis\\demo.mp4'
path_save_movie = 'C:\\Users\\user\\Desktop\\tenis\\tenis\\output_with_boxes.mp4'

cap = cv2.VideoCapture(path_movie)
width_movie = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_movie = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(path_save_movie, fourcc, fps, (width_movie, height_movie))


def extract_features(landmarks):
    features = []
    for i in range(33):
        lm = landmarks.landmark[i]
        features.extend([
            lm.x,
            lm.y,
            lm.z
        ])

    while len(features) < 181:
        features.append(0.0)

    return features


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        x_min = int(min([lm.x for lm in results.pose_landmarks.landmark]) * width_movie)
        y_min = int(min([lm.y for lm in results.pose_landmarks.landmark]) * height_movie)
        x_max = int(max([lm.x for lm in results.pose_landmarks.landmark]) * width_movie)
        y_max = int(max([lm.y for lm in results.pose_landmarks.landmark]) * height_movie)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        features = extract_features(results.pose_landmarks)

        prediction = model.predict([features])[0]

        action_dict = {0: "Idle", 1: "Forehand", 2: "Backhand", 3: "Foreslice", 4: "Backslice"}
        action_name = action_dict.get(prediction, "Unknown")

        cv2.putText(frame, f"Action: {action_name}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Pose Detection with Action Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video başarıyla kaydedildi: {path_save_movie}")
