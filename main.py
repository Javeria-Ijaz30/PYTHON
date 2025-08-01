import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

# Load encodings
with open('encodings/encodings.pickle', 'rb') as f:
    known_encodings = pickle.load(f)

known_face_encodings = known_encodings['encodings']
known_face_names = known_encodings['names']

attendance_file = 'attendance/attendance.csv'

def mark_attendance(name):
    now = datetime.now()
    time = now.strftime('%H:%M:%S')
    date = now.strftime('%Y-%m-%d')
    entry = f"{name},{date},{time}"
    with open(attendance_file, 'a') as f:
        f.write(f"{entry}\n")

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press Q to exit.")

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for face_encoding, face_location in zip(encodings, faces):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
