import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def plot_cuboid(ax, origin, size, color='cyan', alpha=0.3, edge_color='k'):
    """
    원래 좌표계 origin: [x, y, z]와 size: [dx, dy, dz]를
    새로운 좌표계로 변환:
       - 수평(ground) 평면: (y, z)
       - 수직(높이): x
    즉, new_origin = [origin[1], origin[2], origin[0]]
         new_size = [size[1], size[2], size[0]]
    그리고 변환된 좌표계에서 직육면체를 그림.
    """
    new_origin = [origin[1], origin[2], origin[0]]
    new_size    = [size[1], size[2], size[0]]
    x, y, z = new_origin
    dx, dy, dz = new_size

    # 8개 꼭짓점 계산
    vertices = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz]
    ])


    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 바닥면
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 천장면
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 한쪽 측면
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 반대 측면
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 오른쪽 측면
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # 왼쪽 측면
    ]

    poly3d = Poly3DCollection(faces, facecolors=color, edgecolors=edge_color, linewidths=0.1, alpha=alpha)
    ax.add_collection3d(poly3d)

# 복셀 타입에 따른 기능표
func = {
    -1: "None",
     0: "lobby, corridor",
     1: "restroom",
     2: "stairs",
     3: "elevator",
     4: "office",
     5: "mechanical"
}

# 복셀 타입에 따른 색상 매핑
color_mapping = {
    -1: "gray",
     0: "yellow",
     1: "lightcoral",
     2: "lightcoral",
     3: "lightcoral",
     4: "skyblue",
     5: "lightcoral"
}

# 업로드한 복셀 그래프 파일 읽기 (예: "voxel_000000.json")
# with open("Data/6types-raw_data/voxel_data/voxel_100000.json", "r") as f:
with open(f"Data/6types-output/voxel_data/voxel_g.json", "r") as f:
    # with open("inference/Mar03_18-05-25_0/320_Mar04_09-34-27/var_output2/voxel_data/voxel_000000.json", "r") as f:
    voxel_data = json.load(f)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

    # 각 복셀을 시각화 (텍스트 주석은 제거)
for voxel in voxel_data["voxel_node"]:
    if voxel["type"] != -1:
        origin = voxel["coordinate"]    # 예: [0.0, 4.0, 15.0]
        size   = voxel["dimension"]       # 예: [7.0, 6.0, 3.0]
        voxel_type = voxel["type"]
            
        color = color_mapping.get(voxel_type, "white")
        plot_cuboid(ax, origin, size, color=color, alpha=0.8)



# 범례 추가
legend_elems = [
    Patch(facecolor="skyblue", label='Lobby / Corridor'),
    Patch(facecolor="lightcoral", label='core (Stair·Elev·WC·M/E)'),
    Patch(facecolor="violet", label='Office')
]
ax.legend(handles=legend_elems, loc='upper left')


# 축 레이블: 여기서 수평 평면은 (Y, Z)이고, 수직(높이)는 X (즉, 원래 x가 높이)
ax.set_xlabel("Y")
ax.set_ylabel("Z")
ax.set_zlabel("X (Height)")
plt.title("Voxel Graph Visualization (Ground: YZ Plane)")

# 축 한계 (데이터에 따라 조정)
ax.set_xlim([0, 50])
ax.set_ylim([0, 50])
ax.set_zlim([0, 50])


plt.show()