import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import statistics

# ================== EMOTION MAP ==================
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

# ================== MONGODB SETUP ==================
MONGO_URI = "mongodb+srv://bintangananda405:RF7TIDmpByWYtpHq@moodhistory.sguzz0x.mongodb.net/?retryWrites=true&w=majority&appName=moodHistory"
client = MongoClient(MONGO_URI)

try:
    client.admin.command('ping')
    db = client["moodDatabase"]
    collection = db["moodCollection"]
    mongo_connected = True
except Exception as e:
    st.error(f"Gagal koneksi MongoDB: {e}")
    mongo_connected = False

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Analisa Emosi Mahasiswa", layout="centered")
st.title("📊 Analisa Emosi Mahasiswa Berdasarkan MongoDB")

if mongo_connected:
    data_cursor = collection.find().sort("timestamp", -1).limit(50)
    data = list(data_cursor)

    if data:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", ascending=True)

        # Hitung rata-rata berdasarkan EMOTION_MAP
        emotion_values = [EMOTION_MAP.get(row["emotion"], -1) for _, row in df.iterrows()]
        avg_value = statistics.mean([v for v in emotion_values if v >= 0])

        if avg_value > 0.5:
            status = "✅ Siswa sedang semangat belajar berdasarkan analisa emosi!"
            status_color = "green"
        else:
            status = "⚠️ Siswa sedang kurang semangat belajar berdasarkan analisa emosi."
            status_color = "orange"

        st.markdown(f"<h3 style='color:{status_color};'>{status}</h3>", unsafe_allow_html=True)
        st.markdown("### 🕒 Riwayat Emosi (50 Data Terakhir)")
        st.dataframe(df[["timestamp", "emotion"]].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Belum ada data emosi yang tersimpan di MongoDB.")
else:
    st.stop()
