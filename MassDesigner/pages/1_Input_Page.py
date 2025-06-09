import streamlit as st
import json
import os

st.set_page_config(page_title="Mass Designer", layout="wide")
st.title("🏗️ Mass Designer - 설계 조건 입력")

# -------------------------------
# 1. 대지 조건 입력
# -------------------------------
st.header("📐 대지 정보")

col1, col2 = st.columns(2)
with col1:
    site_width = st.number_input("대지 **세로 길이** (m)", min_value=1.0, value=30.0, step=1.0)
with col2:
    site_depth = st.number_input("대지 **가로 길이** (m)", min_value=1.0, value=40.0, step=1.0)

site_area = site_width * site_depth
st.markdown(f"🔢 **대지 면적 자동 계산:** `{site_area:.2f} ㎡`")

col3, col4 = st.columns(2)
with col3:
    building_coverage = st.slider("건폐율 (%)", min_value=0, max_value=100, value=60) / 100
    st.markdown(f"📏 건폐율: `{building_coverage*100:.0f}%`")
with col4:
    floor_area_ratio = st.slider("용적률 (%)", min_value=0, max_value=1500, value=800) / 100
    st.markdown(f"📐 용적률: `{floor_area_ratio*100:.0f}%`")

col5, col6 = st.columns(2)
with col5:
    height_limit_min = st.number_input("최소 높이 제한 (m)", min_value=0, value=90)
with col6:
    height_limit_max = st.number_input("최대 높이 제한 (m)", min_value=0, value=110)

# -------------------------------
# 2. 요청 사항 입력
# -------------------------------
st.header("✍️ 설계 요청사항 입력")

prompt = st.text_area("GPT에게 설계 방향을 알려주세요!", placeholder="예: 중심 코어형 오피스 매스를 제안해줘...")

# -------------------------------
# 세션 상태 초기화
# -------------------------------
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# -------------------------------
# 3. 생성 버튼
# -------------------------------
if st.button("🚀 Mass Design 생성 시작"):
    # 세션 상태에 정보 저장
    st.session_state.prompt = prompt
    st.session_state.site_width = site_width
    st.session_state.site_depth = site_depth
    st.session_state.site_area = site_area
    st.session_state.building_coverage = building_coverage
    st.session_state.floor_area_ratio = floor_area_ratio
    st.session_state.height_limit = (height_limit_min, height_limit_max)

    st.success("✅ 조건 저장 완료! 다음 단계에서 매스가 생성됩니다.")