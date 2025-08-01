import face_recognition
import os
import cv2
import pickle

dataset_path = 'dataset'
encodings = []
names = []

for user_dir in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user_dir)
    if not os.path.isdir(user_path):
        continue
    for img_name in os.listdir(user_path):
        img_path = os.path.join(user_path, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)
        for encoding in encs:
            encodings.append(encoding)
            names.append(user_dir)

data = {"encodings": encodings, "names": names}
with open("encodings/encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encodings saved.")
