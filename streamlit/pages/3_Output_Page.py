import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
import os

# 기본 설정
st.set_page_config(layout="wide")
st.title("Voxel Mass 시각화 + 규모 검토 (ALT1~3)")

# 복셀 타입에 따른 색상 매핑
color_mapping = {
    0: "#AABECF",   # 로비/복도
    1: "#888888",   # 화장실
    2: "#888888",   # 계단
    3: "#888888",   # 엘리베이터
    4: "#B4A29B",   # 오피스
    5: "#888888"    # 기계실
}

# ALT 경로 설정
alt_paths = [
    ("ALT 1", "Data/6types-output1"),
    ("ALT 2", "Data/6types-output2"),
    ("ALT 3", "Data/6types-output0")
]

# 컬럼 분할
col1, col2 = st.columns([2.5, 1.2])

with col1:
    tab1, tab2, tab3 = st.tabs(["ALT 1", "ALT 2", "ALT 3"])

    for tab, (label, path) in zip([tab1, tab2, tab3], alt_paths):
        with tab:
            # 데이터 로딩
            with open(f"{path}/voxel_data/voxel_g.json", "r") as f:
                voxel_data = json.load(f)
            with open(f"{path}/global_graph_data/global_g.json", "r") as f:
                global_data = json.load(f)

            # 규모 정보 계산
            site_area = global_data["site_area"]
            far = global_data["new_far"]
            total_area = site_area * far * 10

            covered_positions = set()
            for v in voxel_data["voxel_node"]:
                if v["type"] != -1:
                    _, y, z = v["coordinate"]
                    dy, dz = v["dimension"][1], v["dimension"][2]
                    for yi in range(int(y), int(y + dy)):
                        for zi in range(int(z), int(z + dz)):
                            covered_positions.add((yi, zi))
            footprint = len(covered_positions)
            coverage_ratio = footprint / site_area

            office_node = next((n for n in global_data["global_node"] if n["type"] == 4), None)
            efficiency_ratio = office_node.get("new_proportion", 0.0) if office_node else 0.0

            # 요약 텍스트
            st.markdown(f"""
            ### 📊 {label} 규모 요약  
            - **연면적**: {total_area:.2f} ㎡  
            - **건폐율**: {coverage_ratio * 100:.2f} %  
            - **용적률**: {far * 1000:.2f} %  
            - **전용률**: {efficiency_ratio * 100:.2f} %
            """)

            # 시각화
            fig = go.Figure()
            for voxel in voxel_data["voxel_node"]:
                if voxel["type"] == -1:
                    continue
                x, y, z = voxel["coordinate"]
                dx, dy, dz = voxel["dimension"]
                Z = [x, x + dx]
                X = [y, y + dy]
                Y = [z, z + dz]
                vertices = np.array([
                    [X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]],
                    [X[1], Y[1], Z[0]], [X[0], Y[1], Z[0]],
                    [X[0], Y[0], Z[1]], [X[1], Y[0], Z[1]],
                    [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]]
                ])
                i, j, k = [], [], []
                faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]]
                for face in faces:
                    tri1 = [face[0], face[1], face[2]]
                    tri2 = [face[0], face[2], face[3]]
                    i.extend([tri1[0], tri2[0]])
                    j.extend([tri1[1], tri2[1]])
                    k.extend([tri1[2], tri2[2]])
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=i, j=j, k=k,
                    opacity=1,
                    color=color_mapping.get(voxel["type"], "white"),
                    flatshading=True,
                    showscale=False
                ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='Y', yaxis_title='Z', zaxis_title='X',
                    xaxis=dict(range=[0, 70]), yaxis=dict(range=[0, 70]), zaxis=dict(range=[0, 70])
                ),
                scene_aspectmode='manual',
                scene_aspectratio=dict(x=1, y=1, z=1),
                margin=dict(r=0, l=0, b=0, t=30)
            )
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("## 🧾 공간 타입 색상")
    st.markdown("- 🟦 **Lobby / Corridor**")
    st.markdown("- ⚫ **Core (Stair · Elevator · WC · M/E)**")
    st.markdown("- 🟫 **Office**")
