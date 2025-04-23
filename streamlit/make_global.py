# ----------------------------------------------------------------------
# util_graph_global.py   (새 파일로 두거나, 기존 Data/GraphConstructor.py
# 맨 아래쪽에 staticmethod 로 추가해 두시면 됩니다)
# ----------------------------------------------------------------------
from collections import defaultdict
from typing import Dict, Any, List

PROGRAM_CLASS_NUM = 6   # 0~5 : corridor, restroom, stair, elevator, office, mech. room

def construct_global(local_graph: Dict[str, Any],
                     site_area: float,
                     Far: float | None = None) -> Dict[str, Any]:
    """
    local_graph : GraphConstructor.load_graph_jsons 로 읽은 local‑graph(dict 형식)
    site_area   : m² 단위 대지면적
    Far         : (선택) 외부에서 이미 계산된 FAR 값.
                  None 이면 local_graph 의 region_far 합으로 자동 계산.
    -------------------------------------------------------------------
    return      : Building‑GAN 에서 사용하는 global_graph(dict)
                  {
                     "far"        : float,
                     "site_area"  : float,
                     "global_node": [ {...}, {...}, ... ]    # 길이 6
                  }
    """
    # 1) 타입별 면적(=region_far)을 누적
    type_area = defaultdict(float)          # {program_type : 누적 region_far}
    type_conn: dict[int, List[List[int]]] = {t: [] for t in range(PROGRAM_CLASS_NUM)}

    for n in local_graph["node"]:
        t = n["type"]
        type_area[t] += n["region_far"]
        # connection 은 (floor, type, type_id) 로 기록
        type_conn[t].append([n["floor"], n["type"], n["type_id"]])

    # 2) 전체 FAR 계산
    far_total = sum(type_area.values())
    far_value = Far if Far is not None else far_total

    # 3) global_node 리스트 생성
    global_node = []
    for t in range(PROGRAM_CLASS_NUM):
        area_t   = type_area.get(t, 0.0)
        proportion = area_t / far_total if far_total > 0 else 0.0
        global_node.append({
            "type"       : t,
            "proportion" : proportion,      # 해당 타입의 면적 비율
            "connection" : type_conn[t]     # local‑graph 상 대응 노드들
        })

    # 4) 최종 global_graph 반환
    return {
        "far"        : far_value,
        "site_area"  : site_area,
        "global_node": global_node
    }
