import numpy as np
import json
from collections import defaultdict

############################################
# 1) 스케일링 (Width, Length)로 노드 (y,x) 배치
############################################
def scale_and_translate_nodes(local_graph, base_width, base_length):
    """
    local_graph["node"] 내 모든 node.center의 (y,x)를
    (minY~maxY, minX~maxX) -> (0~base_length, 0~base_width) 범위로
    각각 비례 스케일+평행이동
    """
    minY = min(nd["center"][1] for nd in local_graph["node"])
    maxY = max(nd["center"][1] for nd in local_graph["node"])
    minX = min(nd["center"][2] for nd in local_graph["node"])
    maxX = max(nd["center"][2] for nd in local_graph["node"])

    dy = maxY - minY
    dx = maxX - minX
    if abs(dy) < 1e-9:
        dy = 1.0
    if abs(dx) < 1e-9:
        dx = 1.0

    scale_y = base_length / dy
    scale_x = base_width  / dx

    for nd in local_graph["node"]:
        old_y = nd["center"][1]
        old_x = nd["center"][2]
        nd["center"][1] = (old_y - minY) * scale_y
        nd["center"][2] = (old_x - minX) * scale_x
    return local_graph

############################################
# 2) 분할선 처리 (3m 미만 합치고, 6m 초과 나누기)
############################################
def remove_close_lines(sorted_vals, min_gap=3.0):
    """
    인접 선 간격 < min_gap 이면 앞선에 합침
    ex) [0,2.5,5,10], min_gap=3 => [0,5,10]
    """
    if len(sorted_vals)<2:
        return sorted_vals
    merged = [sorted_vals[0]]
    for i in range(1, len(sorted_vals)):
        prev_line = merged[-1]
        curr_line = sorted_vals[i]
        gap = curr_line - prev_line
        if gap < min_gap:
            # 합치기 => curr_line 스킵
            continue
        else:
            merged.append(curr_line)
    return merged

def expand_large_intervals(sorted_vals, max_gap=6.0):
    """
    인접 선 거리 > max_gap이면 max_gap 단위로 쪼개기
    ex) [10,23], gap=13 => [10,16,22,23] (6+6+1)
    """
    if len(sorted_vals)<2:
        return sorted_vals
    new_list = [sorted_vals[0]]
    for i in range(1, len(sorted_vals)):
        start = new_list[-1]
        end   = sorted_vals[i]
        gap   = end - start
        while gap>max_gap:
            subdiv_pt = start + max_gap
            new_list.append(subdiv_pt)
            start = subdiv_pt
            gap = end - start
        if abs(end - new_list[-1])>1e-9:
            new_list.append(end)
    return new_list

############################################
# 3) manhattan 분할 (y,x)
############################################
def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def nearest_neighbor_midlines(node_list_2d):
    """
    node_list_2d = [(y, x), ...]
    => manhattan 보로노이식 분할선 (y_list, x_list)
    """
    arr = np.array(node_list_2d, dtype=float)
    ys, xs = set(), set()

    for i,(y0,x0) in enumerate(arr):
        (y0,x0) =(round(y0),round(x0))
        min_dist = float('inf')
        min_j = -1
        for j,(y1,x1) in enumerate(arr):
            (y1,x1) =(round(y1),round(x1))
            if i==j: continue
            d = manhattan_distance((y0,x0),(y1,x1))
            if d<min_dist:
                min_dist=d
                min_j=j
        y1,x1 = arr[min_j]
        ys.update([y0, 0.5*(y0+y1)])
        xs.update([x0, 0.5*(x0+x1)])
    # 노드 원본도 추가
    for (yy,xx) in node_list_2d:
        ys.add(yy); xs.add(xx)

    return sorted(ys), sorted(xs)

def intervals_from_sorted(vals):
    return [(vals[i],vals[i+1]) for i in range(len(vals)-1)]

############################################
# 4) 층별 zrange => local_graph 그대로 유지
############################################
def compute_floor_zrange(local_graph):
    """
    예: floor별 노드 center z 중 min값만 사용,
        다음층 minz - 현재층 minz => 높이
        (마지막 층은 이전층과 동일 height)
    """
    from collections import defaultdict
    floor_to_nodes = defaultdict(list)
    for nd in local_graph["node"]:
        floor_to_nodes[nd["floor"]].append(nd)

    floors = sorted(floor_to_nodes.keys())

    floor_zrange = {}
    for i, f in enumerate(floors):
        now_zvals = [n["center"][0] for n in floor_to_nodes[f]]
        if not now_zvals:
            continue
        z0 = min(now_zvals)
        if i != len(floors)-1:
            # 다음층
            f_next = floors[i+1]
            next_zvals = [n["center"][0] for n in floor_to_nodes[f_next]]
            if next_zvals:
                height = min(next_zvals) - z0
            else:
                height = 4.0
        else:
            # 마지막 층
            if i>0:
                height = floor_zrange[floors[i-1]][1]
            else:
                height = 4.0
        if abs(height)<1e-9:
            # 최소 두께
            height=4.0
        floor_zrange[f] = (z0,height)

    return floor_zrange

############################################
# 5) 최종 함수: XY엔 min_gap=3, max_gap=6 / Z는 local_graph 그대로
############################################
def construct_voxel_stack_line_merge_and_split_keeping_z(local_graph,
                                                         base_width=40,
                                                         base_length=20,
                                                         min_gap=3.0,
                                                         max_gap=6.0,
                                                         use_height_limit=False): 
    """
    1) 노드 (y,x) -> [0,base_length],[0,base_width] 스케일
    2) manhattan 분할 -> (y_list,x_list)
    3) remove_close_lines(..., min_gap)
    4) expand_large_intervals(..., max_gap)
    5) intervals_from_sorted
    6) compute_floor_zrange_same_as_local => extrude => neighbor
    => z방향은 local_graph 그대로 유지
    => xy방향은 3m미만 합치고, 6m초과 나누는 로직
    """
    # A. 노드 스케일링
    local_graph = scale_and_translate_nodes(local_graph,
                                           base_width, base_length)

    # B. 전체 노드 (y,x) 모아서 manhattan 분할
    all_2d=[]
    for nd in local_graph["node"]:
        cy,cx= nd["center"][1], nd["center"][2]
        all_2d.append((cy,cx))

    if len(all_2d)<2:
        y_list=[0, base_length]
        x_list=[0, base_width]
    else:
        y_list,x_list= nearest_neighbor_midlines(all_2d)
        if len(y_list)<2: y_list=[0, base_length]
        if len(x_list)<2: x_list=[0, base_width]

    # C. 3m 미만 합치기, 6m 초과 나누기, 다시 합치기
    # y_merged = remove_close_lines(y_list, min_gap=min_gap)
    # x_merged = remove_close_lines(x_list, min_gap=min_gap)

    # y_final = expand_large_intervals(y_merged, max_gap=max_gap)
    # x_final = expand_large_intervals(x_merged, max_gap=max_gap)

    # y_final = remove_close_lines(y_final, min_gap=min_gap-1)
    # x_final = remove_close_lines(x_final, min_gap=min_gap-1)

    # C. 나누고 합치기

    y_final = expand_large_intervals(y_list, max_gap=max_gap)
    x_final = expand_large_intervals(x_list, max_gap=max_gap)

    y_final = remove_close_lines(y_final, min_gap=min_gap)
    x_final = remove_close_lines(x_final, min_gap=min_gap)

    # y_final = y_list
    # x_final = x_list





    # 정렬
    y_final = sorted(set(y_final))
    x_final = sorted(set(x_final))

    y_intervals = intervals_from_sorted(y_final)
    x_intervals = intervals_from_sorted(x_final)

    # D. 층별 zrange, local_graph 그대로 유지
    floor_zrange= compute_floor_zrange(local_graph)
    floors = sorted(floor_zrange.keys())


    # E. extrude
    total_area = 0
    voxel_nodes=[]
    for f in floors:
        (zmin,zmax)= floor_zrange[f]
        dz= zmax
        for row,(yA,yB) in enumerate(y_intervals):
            dy= yB-yA
            if dy<=1e-9: 
                continue
            for col,(xA,xB) in enumerate(x_intervals):
                dx= xB-xA
                if dx<=1e-9:
                    continue
                area = dx * dy
                total_area += area
                voxel_nodes.append({
                    "location":[f,row,col],
                    "type":-1,
                    "type_id":0,
                    "coordinate":[zmin,yA,xA], # (z,y,x)
                    "dimension":[dz,dy,dx],
                    "weight":area,
                    "__raw_area": area,      # 임시 저장
                    "neighbors":[]
                })
    
    if total_area > 0:
        for v in voxel_nodes:
            v["weight"] = v["__raw_area"] / total_area
            del v["__raw_area"]
    else:
        for v in voxel_nodes:
            v["weight"] = 0.0
    

    # F. neighbor 연결
    loc2idx={}
    for i,v in enumerate(voxel_nodes):
        loc2idx[tuple(v["location"])]=i
    directions= [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]
    for i,v in enumerate(voxel_nodes):
        f,r,c= v["location"]
        nb=[]
        for df,dr,dc in directions:
            nf,nr,nc= (f+df, r+dr, c+dc)
            if (nf,nr,nc) in loc2idx:
                nb.append([nf,nr,nc])
        v["neighbors"]=nb

    # 디버그 검사
    debug_check_node_in_voxel(voxel_nodes, local_graph)

        # 높이 사선 제한 적용 (옵션)
    voxel_nodes = apply_north_height_setback(voxel_nodes, use_limit=use_height_limit)

    return convert_numpy({"voxel_node": voxel_nodes})


############################################
# 디버그 검사
############################################
def debug_check_node_in_voxel(voxel_nodes, local_graph):
    from collections import defaultdict
    floor_to_nodes= defaultdict(list)
    for nd in local_graph["node"]:
        floor_to_nodes[nd["floor"]].append(nd)

    floor_vox_map= defaultdict(list)
    for i,v in enumerate(voxel_nodes):
        f,r,c= v["location"]
        z0,y0,x0= v["coordinate"]
        dz,dy,dx= v["dimension"]
        z1,y1,x1= z0+dz, y0+dy, x0+dx
        floor_vox_map[f].append((z0,z1, y0,y1, x0,x1, i))

    for f, nds in floor_to_nodes.items():
        if f not in floor_vox_map:
            continue
        for nd in nds:
            z,y,x= nd["center"]
            found=False
            for (Z0,Z1,Y0,Y1,X0,X1,vid) in floor_vox_map[f]:
                if Z0<=z<=Z1 and Y0<=y<=Y1 and X0<=x<=X1:
                    found=True
                    break
            if not found:
                print(f"[WARN] Node center={nd['center']} floor={f} not in any voxel")

def convert_numpy(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k:convert_numpy(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(e) for e in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def apply_north_height_setback(voxel_nodes, use_limit=True):
    if not use_limit:
        return voxel_nodes  # 제한 미사용 시 그대로 반환

    filtered = []
    for v in voxel_nodes:
        z0, y0, _ = v["coordinate"]
        dz, dy, dx = v["dimension"]
        z_top = z0 + dz

        if z_top <= 15.0:
            # 10m 이하면 제한 없이 통과
            filtered.append(v)
            continue

        setback = z_top * 0.5
        if y0 >= setback:  # 사선 제한 충족 시만 유지
            filtered.append(v)

    # neighbor 재연결
    loc2idx = {tuple(v["location"]): i for i, v in enumerate(filtered)}
    dirs = [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]

    for v in filtered:
        f,r,c = v["location"]
        nb = []
        for df,dr,dc in dirs:
            key = (f+df, r+dr, c+dc)
            if key in loc2idx:
                nb.append(list(key))
        v["neighbors"] = nb

    return filtered



def pre_label_program_nodes(voxel_graph, local_graph):
    """
    voxel_graph["voxel_node"] 각 복셀과,
    local_graph["node"] 각 노드( floor, center[z,y,x], type )를 비교
    => 노드가 들어가는 복셀 찾아 type 지정
    base_width/base_length는 voxel_graph에서 실제 extents를 계산해서 사용
    """
    voxel_nodes = voxel_graph["voxel_node"]

    # (0) voxel_graph 전체 y,x 범위(실제 좌표) 파악 → base_length, base_width 계산
    y0_list = [vx["coordinate"][1]                for vx in voxel_nodes]
    y1_list = [vx["coordinate"][1] + vx["dimension"][1] for vx in voxel_nodes]
    x0_list = [vx["coordinate"][2]                for vx in voxel_nodes]
    x1_list = [vx["coordinate"][2] + vx["dimension"][2] for vx in voxel_nodes]

    voxel_minY, voxel_maxY = min(y0_list), max(y1_list)
    voxel_minX, voxel_maxX = min(x0_list), max(x1_list)

    base_length = voxel_maxY - voxel_minY  # Y축 전체 길이
    base_width  = voxel_maxX - voxel_minX  # X축 전체 길이
    if base_length == 0: base_length = 1.0
    if base_width  == 0: base_width  = 1.0

    # (1) local_graph에서 raw y,x 범위 파악
    y_list = [nd["center"][1] for nd in local_graph["node"]]
    x_list = [nd["center"][2] for nd in local_graph["node"]]
    minY, maxY = min(y_list), max(y_list)
    minX, maxX = min(x_list), max(x_list)

    dy = maxY - minY or 1.0
    dx = maxX - minX or 1.0

    # (2) 각 프로그램 노드마다,
    for nd in local_graph["node"]:
        f = nd["floor"]
        zc = nd["center"][0]
        raw_y = nd["center"][1]
        raw_x = nd["center"][2]
        node_type = nd["type"]

        # 2-1) local 좌표 → [0,1] 정규화
        rel_y = (raw_y - minY) / dy
        rel_x = (raw_x - minX) / dx

        # 2-2) voxel_graph 실제 좌표계로 매핑
        norm_y = voxel_minY + rel_y * base_length
        norm_x = voxel_minX + rel_x * base_width

        # (3) 해당 floor(f)에 있는 voxel 중에서
        found = False
        for vx in voxel_nodes:
            if vx["location"][0] != f:
                continue
            z0, y0, x0 = vx["coordinate"]
            dz, dy_, dx_ = vx["dimension"]
            z1, y1, x1 = z0 + dz, y0 + dy_, x0 + dx_

            # 물리 좌표(zc, norm_y, norm_x)가 속하면 라벨 지정
            if (z0 <= zc <= z1) and (y0 <= norm_y <= y1) and (x0 <= norm_x <= x1):
                vx["type"] = node_type
                found = True
                break

        if not found:
            print(f"[WARN] Node {nd} not found in any voxel.")




############################################
# 사용 예시
############################################
if __name__=="__main__":
    with open("Data/6types-raw_data/local_graph_data/graph_local_022499.json","r") as f:
        local_graph = json.load(f)

    # XY는 (base_width=40, base_length=20)로 스케일
    # 3m 미만이면 합치고, 6m 초과면 나누기
    # Z는 local_graph 노드의 min/max 그대로
    voxel_graph = construct_voxel_stack_line_merge_and_split_keeping_z(
        local_graph,
        base_width=25,
        base_length=35,
        min_gap=3,
        max_gap=6,
        use_height_limit=False 
    )
    #  2, 8가 뭔가 적당한 느낌

    # pre_label_program_nodes(voxel_graph, local_graph)  # 복셀이 클수록 좀 더 잘됨

    with open("Data/6types-raw_data/voxel_data/voxel_022499.json","w") as wf:
        json.dump(voxel_graph, wf, indent=2)
