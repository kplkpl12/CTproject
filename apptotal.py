import streamlit as st
import streamlit.components.v1 as htmlviewr
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---- HTML 파일 불러오기 ----
def load_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


html1 = load_html("ct1_2.html")
html2 = load_html("ct2_5.html")
html3 = load_html("ct3_1.html")

# ---- 기본 지식 데이터 ----
default_knowledge = {
    "사과": ["C", "H", "O"],
    "나무": ["C", "H", "O"],
    "돌": ["O", "Si", "Li", "Al", "B"],
    "물": ["H", "O"],
    "구름": ["H", "O"],
    "혈액": ["O", "H", "C", "N", "Na", "Ca", "P"],
    "우유": ["C", "H", "O", "N", "Ca", "P"],
    "소금": ["Na", "Cl"],
    "모래": ["Si", "O"],
    "생물": ["C", "H", "O", "N"],
}

# ---- 학습용 이미지 저장 경로 ----
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_CSV = os.path.join(DATA_DIR, "train_data.csv")
if not os.path.exists(DATA_CSV):
    pd.DataFrame(columns=["filename", "label", "elements"]).to_csv(
        DATA_CSV, index=False
    )


# ---- 텍스트 유사도 기반 분류 ----
def find_most_similar_label(input_label, existing_labels):
    vectorizer = TfidfVectorizer().fit_transform([input_label] + existing_labels)
    sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_match_index = np.argmax(sim)
    return existing_labels[best_match_index]


# ---- 메인 화면 ----
st.set_page_config(layout="wide")
st.title("Juhee's CT 문제 & AI 구성원소 예측 앱")

col1, col2 = st.columns((2, 2))

with col1:
    st.header("💡 CT 문제 모음")
    with st.expander("CT Problem1"):
        htmlviewr.html(html1, height=800)
    with st.expander("CT Problem2"):
        htmlviewr.html(html2, height=800)
    with st.expander("CT Problem3"):
        htmlviewr.html(html3, height=1200)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<font color='BLUE'>(c)copyright. all rights reserved by skykang</font>",
    unsafe_allow_html=True,
)
