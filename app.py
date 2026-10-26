"""Aplikasi interaktif. Jalankan: streamlit run app.py"""

import pandas as pd
import streamlit as st

from model_tidur import kategori, latih_model, prediksi

st.set_page_config(page_title="Simulasi Medsos dan Kualitas Tidur")


@st.cache_resource
def muat_model():
    return latih_model()


model, (jam_min, jam_maks), akurasi = muat_model()

st.title("Simulasi: Jam Medsos dan Kualitas Tidur")
st.write("Geser slider untuk melihat tebakan model tentang skor kualitas tidur (1 = buruk, 5 = sangat baik).")

jam = st.slider(
    "Jam penggunaan medsos per hari",
    min_value=round(jam_min, 1),
    max_value=round(jam_maks, 1),
    value=4.0,
    step=0.1,
)

tebakan, peluang, perkiraan = prediksi(model, jam)

kolom1, kolom2 = st.columns(2)
kolom1.metric("Tebakan skor tidur", f"{tebakan}  ({kategori(tebakan)})")
kolom2.metric("Perkiraan rata-rata", f"{perkiraan:.2f}")

st.subheader("Peluang tiap skor menurut model")
st.bar_chart(pd.Series({str(k): v for k, v in peluang.items()}))

st.info(
    f"Akurasi model (k-fold): {akurasi:.2f}. Tebakan asal hanya sekitar 0,34. "
    "Artinya model lebih baik dari asal tebak, tetapi hampir separuh tebakan masih meleset."
)
st.caption(
    "Batasan: hasil menunjukkan hubungan pada dataset ini, bukan sebab-akibat. "
    "Data berasal dari jawaban sendiri dan asal-usulnya tidak terverifikasi. "
    "Ini simulasi untuk tugas, bukan alat diagnosis."
)
