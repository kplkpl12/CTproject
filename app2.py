import streamlit as st
import os
import shutil
from PIL import Image
import pandas as pd
import base64
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 기본 데이터셋
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

# 이미지 저장 경로
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 학습 데이터 CSV
DATA_CSV = os.path.join(DATA_DIR, "train_data.csv")
if not os.path.exists(DATA_CSV):
    pd.DataFrame(columns=["filename", "label", "elements"]).to_csv(
        DATA_CSV, index=False
    )


# 텍스트 유사도 기반 간단한 분류
def find_most_similar_label(input_label, existing_labels):
    vectorizer = TfidfVectorizer().fit_transform([input_label] + existing_labels)
    sim = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_match_index = np.argmax(sim)
    return existing_labels[best_match_index]


# Streamlit UI
st.title("📷 사진 속 물체 구성 원소 예측 AI")

tab1, tab2 = st.tabs(["🔧 학습하기", "🔍 테스트하기"])

with tab1:
    st.header("🔧 새로운 사진과 원소를 학습시켜보세요!")
    uploaded = st.file_uploader(
        "이미지를 업로드하세요", type=["png", "jpg", "jpeg"], key="train"
    )
    label = st.text_input("이 물체의 이름은?")
    elements = st.text_input(
        "이 물체를 구성하는 주요 원소는? (쉼표로 구분, 예: C, H, O)"
    )

    if uploaded and label and elements:
        save_path = os.path.join(DATA_DIR, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.read())
        new_row = pd.DataFrame(
            [[uploaded.name, label, elements]],
            columns=["filename", "label", "elements"],
        )
        df = pd.read_csv(DATA_CSV)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_CSV, index=False)
        st.success("✅ 학습 데이터가 저장되었습니다!")

with tab2:
    st.header("🔍 사진을 분석해 구성 원소를 알려드릴게요")
    test_image = st.file_uploader(
        "테스트할 이미지를 업로드하세요", type=["png", "jpg", "jpeg"], key="test"
    )

    if test_image:
        image = Image.open(test_image)
        st.image(image, caption="업로드한 이미지", use_column_width=True)

        # label 추론을 위해 파일명에서 추정하거나 기본 분류 적용
        name_guess = os.path.splitext(test_image.name)[0]
        st.info(f"📌 예상되는 이름: **{name_guess}**")

        # 1단계: 기본 지식 기반 검색
        matched = None
        for key in default_knowledge:
            if key in name_guess:
                matched = key
                break

        if matched:
            elements = default_knowledge[matched]
            st.success(f"🧪 예측된 원소: {', '.join(elements)} (기본 지식 기반)")
        else:
            # 2단계: 기존 학습 데이터 기반 유사도 매칭
            df = pd.read_csv(DATA_CSV)
            if df.empty:
                st.warning(
                    "학습된 데이터가 없어요! 먼저 학습탭에서 데이터를 추가해주세요."
                )
            else:
                best_match = find_most_similar_label(name_guess, df["label"].tolist())
                elements = df[df["label"] == best_match]["elements"].values[0]
                st.success(f"🔍 유사한 학습 예시로부터 예측됨: {best_match}")
                st.success(f"🧪 예측된 원소: {elements}")
