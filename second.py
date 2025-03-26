import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import cv2
import os
import mediapipe as m
import joblib
import numpy as np


backhand = pd.read_csv('backhand.csv')
backslice = pd.read_csv('backslice.csv')
forehand = pd.read_csv('forehand.csv')
foreslice = pd.read_csv('foreslice.csv')

train_data = pd.concat([backhand, backslice, forehand, foreslice], ignore_index=True)

test_data = pd.read_csv('test.csv')

X = train_data.drop('action_gt_num', axis=1)
y = train_data['action_gt_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

joblib.dump(model, 'tennis_action_model.pkl') #Model Accuracy: 0.95397744590064


import cv2
import mediapipe as mp
import joblib
import numpy as np

# Modeli yükle
model = joblib.load('tennis_action_model.pkl')

# Pose modelini başlat
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video yolları
input_video_path = "demo.mp4"
output_video_path = "output.mp4"

# Video giriş ve çıkış ayarları
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if fps == 0:  # Hatalı FPS değerini düzelt
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

def extract_features(landmarks):
    features = []
    for i in range(33):  # Mediapipe 33 poz noktası üretir
        lm = landmarks.landmark[i]
        features.extend([lm.x, lm.y, lm.z])

    # Boş veri eklemeyi önlemek için
    while len(features) < model.n_features_in_:
        features.append(0.0)

    return features


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        features = extract_features(results.pose_landmarks)
        prediction = model.predict([features])[0]

        action_dict = {
            0: "Idle",
            1: "Forehand",
            2: "Backhand",
            3: "Foreslice",
            4: "Backslice"
        }
        action_name = action_dict.get(prediction, "Unknown")

        cv2.putText(frame, f"Action: {action_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tennis Action Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video başarıyla kaydedildi: {output_video_path}")

