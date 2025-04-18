import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import time
from collections import deque
import statistics
import requests

# ================= UBIDOTS CONFIG =================
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
            print(f"[UBIDOTS] Emosi '{emotion}' berhasil dikirim dengan data tambahan.")
        else:
            print(f"[UBIDOTS] Gagal kirim: {response.text}")
    except Exception as e:
        print(f"[UBIDOTS] Error: {e}")


# ================= EMOTION CONFIG =================
emotion_colors = {
    'angry': '#FF4B4B', 'happy': '#FFD700', 'sad': '#87CEEB', 'neutral': '#D3D3D3',
    'fear': '#9370DB', 'surprise': '#FFA500', 'disgust': '#98FB98', 'unknown': '#FFFFFF'
}

emotion_emojis = {
    'angry': '😠', 'happy': '😄', 'sad': '😢', 'neutral': '😐',
    'fear': '😱', 'surprise': '😲', 'disgust': '🤢', 'unknown': '❓'
}

# ================= STREAMLIT CONFIG =================
st.set_page_config(page_title="Deteksi Emosi Mahasiswa", layout="centered")
st.title("🎭 Deteksi Emosi Mahasiswa Real-Time")

ESP32_URL = "http://192.168.166.31:81/stream"

if 'run' not in st.session_state:
    st.session_state.run = False
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=30)

start_button = st.button("Mulai Deteksi" if not st.session_state.run else "Hentikan Deteksi")
if start_button:
    st.session_state.run = not st.session_state.run
    if not st.session_state.run:
        st.session_state.emotion_history.clear()

frame_placeholder = st.empty()
text_placeholder = st.empty()

if st.session_state.run:
    cap = cv2.VideoCapture(ESP32_URL)

    if not cap.isOpened():
        st.error("❌ Tidak bisa membuka stream dari ESP32-CAM.")
    else:
        st.success("✅ Stream ESP32-CAM berhasil dibuka.")

        last_sent_time = 0
        interval_send = 5  # Kirim ke Ubidots tiap 5 detik

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Tidak bisa membaca frame dari ESP32.")
                break

            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion_scores = result[0]['emotion']
                st.session_state.emotion_history.append(emotion_scores)
            except:
                continue

            if st.session_state.emotion_history:
                avg_emotions = {}
                for em in st.session_state.emotion_history[0].keys():
                    avg_emotions[em] = statistics.mean([e[em] for e in st.session_state.emotion_history])
                dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            else:
                dominant_emotion = 'unknown'

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            bg_color = emotion_colors.get(dominant_emotion, '#FFFFFF')
            emoji = emotion_emojis.get(dominant_emotion, '❓')

            frame_placeholder.image(img_pil, channels="RGB", use_container_width=True)
            text_placeholder.markdown(
                f"<h2 style='text-align: center; color: black; background-color:{bg_color}; padding: 1rem;'>"
                f"Emosi Dominan (Rata-rata): {dominant_emotion.upper()} {emoji}</h2>",
                unsafe_allow_html=True
            )

            # Kirim emosi ke Ubidots tiap beberapa detik
            if time.time() - last_sent_time > interval_send:
                kirim_emosi_ke_ubidots(dominant_emotion)
                last_sent_time = time.time()

            time.sleep(0.05)

        cap.release()
        frame_placeholder.empty()
        text_placeholder.empty()
