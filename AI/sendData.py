import cv2
import time
import statistics
from collections import deque
from deepface import DeepFace
from pymongo.mongo_client import MongoClient
from datetime import datetime
import requests
import numpy as np
from mtcnn import MTCNN

UBIDOTS_TOKEN = "BBUS-NkkUJPZOo48KrOvlHj7b1RnCTkWBZZ"
DEVICE_NAME = "esp32_cam"
EMOTION_MAP = {
    "angry": 0.1,
    "disgust": 0.2,
    "fear": 0.3,
    "sad": 0.4,
    "neutral": 0.5,
    "surprise": 0.8,
    "happy": 1.0,
    "unknown": -1
}

def kirim_emosi_ke_ubidots(emotion):
    value = EMOTION_MAP.get(emotion, -1)
    url = f"https://industrial.api.ubidots.com/api/v1.6/devices/{DEVICE_NAME}/"
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "emotion": {
            "value": value,
            "context": {"label": emotion}
        },
        emotion: {
            "value": 1
        }
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            print(f"[UBIDOTS] Emosi '{emotion}' berhasil dikirim.")
        else:
            print(f"[UBIDOTS] Gagal kirim: {response.text}")
    except Exception as e:
        print(f"[UBIDOTS] Error: {e}")

MONGO_URI = "mongodb+srv://bintangananda405:RF7TIDmpByWYtpHq@moodhistory.sguzz0x.mongodb.net/?retryWrites=true&w=majority&appName=moodHistory"
client = MongoClient(MONGO_URI)
try:
    client.admin.command('ping')
    print("[MongoDB] Terhubung ke MongoDB.")
    mongo_connected = True
except Exception as e:
    print(f"[MongoDB] Gagal koneksi: {e}")
    mongo_connected = False

def kirim_emosi_ke_mongodb(emotion, timestamp):
    if not mongo_connected:
        return
    try:
        db = client["moodDatabase"]
        collection = db["moodCollection"]
        doc = {
            "emotion": emotion,
            "timestamp": timestamp.isoformat()
        }
        collection.insert_one(doc)
        print(f"[MongoDB] Emosi '{emotion}' berhasil disimpan.")
    except Exception as e:
        print(f"[MongoDB] Gagal simpan emosi: {e}")

ESP32_URL = 0 # Trial dengan webcam
cap = cv2.VideoCapture(ESP32_URL)
if not cap.isOpened():
    print("❌ Tidak bisa membuka webcam.")
    exit()

print("✅ Webcam berhasil dibuka.")

detector = MTCNN()

emotion_history = deque(maxlen=30)
last_sent_time = 0
interval_send = 5

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    resized = cv2.resize(sharpened, (224, 224))
    return resized

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Tidak bisa membaca frame dari webcam.")
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            face_crop = frame[y:y+h, x:x+w]

            preprocessed_face = preprocess_image(face_crop)

            try:
                result = DeepFace.analyze(preprocessed_face, actions=['emotion'], enforce_detection=False)
                emotion_scores = result[0]['emotion']
                emotion_history.append(emotion_scores)
            except Exception as e:
                print(f"[DeepFace] Gagal deteksi: {e}")
                continue

        if emotion_history:
            avg_emotions = {
                em: statistics.mean([e[em] for e in emotion_history])
                for em in emotion_history[0].keys()
            }
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
        else:
            dominant_emotion = 'unknown'

        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"{dominant_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Deteksi Emosi Webcam', frame)

        if time.time() - last_sent_time > interval_send:
            timestamp_now = datetime.now()
            kirim_emosi_ke_ubidots(dominant_emotion)
            kirim_emosi_ke_mongodb(dominant_emotion, timestamp_now)
            last_sent_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n⏹️ Proses dihentikan oleh user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam ditutup.")
