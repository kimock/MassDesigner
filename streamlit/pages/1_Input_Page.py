import streamlit as st
import json
import os

st.title("Mass Designer - 초기 조건 입력")

# 1. 대지 조건 입력
st.header("📐 대지 조건 입력")

col1, col2 = st.columns(2)
with col1:
    site_width = st.number_input("대지 세로 (m)", min_value=1.0, value=20.0)
with col2:
    site_depth = st.number_input("대지 가로 (m)", min_value=1.0, value=40.0)

site_area = site_width * site_depth
st.markdown(f"🧮 **자동 계산된 대지면적: {site_area:.2f} ㎡**")

building_coverage = st.slider("건폐율 (%)", min_value=0, max_value=100, value=60)/100
floor_area_ratio = st.slider("용적률 (%)", min_value=0, max_value=1500, value=800)/ 100

col3, col4 = st.columns(2)
with col3:
    height_limit_min = st.number_input("최소 높이 제한 (m)", min_value=0, value=90)
with col4:
    height_limit_max = st.number_input("최대 높이 제한 (m)", min_value=0, value=110)


# 2. 프로그램 그래프 업로드
st.header("📁 프로그램 그래프 업로드")
uploaded_local_graph = st.file_uploader("Local Graph (JSON)", type=["json"])



# 상태 변수
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None

# 3. Generate 버튼
if st.button("Generate Mass Design"):
    if uploaded_local_graph is not None:
        local_graph = json.load(uploaded_local_graph)
        with open("Data/6types-raw_data/local_graph_data/local_g.json","w") as wf:
            json.dump(local_graph, wf, indent=2)
        st.session_state.graph_data = local_graph  # 세션에 저장

         # 세션 상태 저장
        st.session_state.graph_data = local_graph
        st.session_state.site_width = site_width
        st.session_state.site_depth = site_depth
        st.session_state.site_area = site_area
        st.session_state.building_coverage = building_coverage
        st.session_state.floor_area_ratio = floor_area_ratio
        st.session_state.height_limit = (height_limit_min, height_limit_max)


        st.success("그래프 업로드 및 조건 입력 완료! 다음 단계로 이동합니다.")

    else:
        st.warning("프로그램 그래프 파일을 업로드해주세요.")

st.markdown("[👉 다음 단계로 이동](2_Generation_Page)", unsafe_allow_html=True)