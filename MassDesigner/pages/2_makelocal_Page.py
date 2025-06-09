from openai import OpenAI
import streamlit as st
import json
import os
from dotenv import load_dotenv
import re


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

SYSTEM_1 ="""
당신은 건축 매스 디자인 전문가입니다.
아래 “라벨링 원칙”에 따라, 사용자의 자연어 요청을 해석하고 건축적 판단을 거쳐서, **기본 구조 템플릿**에 맞춘 라벨 요약문만 출력하세요. 절대 다른 설명이나 주석을 붙이지 마십시오.

<— 라벨링 원칙 —>
1. 층별 높이는 “1층”, “2층” 등으로 표기.
2. 한 층 내 위치는 “동쪽”, “서쪽”, “남쪽”, “북쪽”, “중앙”, "북동쪽","남동쪽", "북서쪽", "남서쪽" 등의 상대적 방향 용어로만 표현.
3. 코어(restroom, stair, elevator, mechanical)는 채광·진입 동선을 고려해 배치해야한다.
4. 오피스, 복도, 로비, 코어 등 기능을 명확히 구분.
5. 연결 관계는 “연결되어 있다”로 서술.
6. 돌출부나 테라스는 “돌출되어 …가 된다”로 서술.
7. 전체 용적률(FAR), 프로그램 비율(TPR) 등 필요한 제약은 한 문장으로 간략히 추가 가능.
8. 오피스는 채광과 조망이 가장 좋은 장소에 배치합니다, 코어는 채광이 별로 좋지 않는 위치에 배치합니다.
9. 로비와 복도는 별도로 분리해서 생각하세요.

<— 기본 구조 템플릿 —＞
[층수] 규모의 (형용사) 오피스 매스를 설계해줘.
[위치]에 [타입]과 [타입]이 있으며,
[위치]에 [타입]이 배치되어야 한다.
[코어]는 [위치]에 있다.
[층수(최고 층수 제외)]의 [오피스 or 로비]은 [위치]로 돌출되어서 [테라스]가 된다.
[1층]의 [위치]는 [필로티] 형태이다.

— 출력 규칙 —
• 오직 위 템플릿 형식의 라벨 요약문만 출력.
• 문장 순서·형식은 템플릿을 따르되, 사용자 요청에 맞춰 자연스럽게 서술.
• 추가 설명·주석 절대 금지.

<예시>
-6층 규모의 매스를 설계해줘, 동쪽에 복도가 있다. 복도의 북쪽과 남쪽 부분에는 엘리베이터와 화장실이 있으며, 각각 북쪽과 남쪽의 복도와 연결되어 있다. 복도의 북서쪽에는 계단실이 있다. 매스의 남쪽과 남서쪽은 오피스가 있으며, 남쪽의 오피스는 동쪽의 복도와 연결되어 있다. 6층의 남쪽 오피스는 북쪽의 계단실과도 연결되어 있다. 1층의 로비는 남쪽에 있다.
-11층 규모의 매스를 설계해줘, 서쪽에 코어가 있다. 북서쪽에 엘리베이터가 있고, 남서쪽에 기계실과 화장실이 있으며, 홀은 서쪽에서 기능실을 연결한다. 계단실은 중앙에 있으며 홀과 연결되어있다. 오피스는 동쪽에 위치해있다. 1층 로비는 동쪽으로 돌출되어 있으며, 1층 오피스는 동남쪽으로 돌출되어 테라스가 된다.
-7층 규모의 매스를 설계해줘, 북서쪽과 남서쪽에 엘리베이터와 계단이 있으며, 기계실이 북쪽과 북서쪽에 있다. 복도는 중앙에서 살짝 서쪽에 있으며, 복도의 서쪽에는 화장실이 있다. 오피스는 동쪽에 있으며, 2층부터4층까지 오피스는 동쪽으로 돌출되어 테라스가 된다.


"""

SYSTEM = """
당신은 건축 매스 디자인을 위한 local_graph 생성 전문가입니다. 
사용자의 자연어 요청을 해석해, 반드시 아래 스키마에 맞춘 **유효한 JSON**만을 출력하세요. 절대 일반 텍스트나 추가 설명을 붙이지 마십시오.

<— Local_graph 스키마 —>
{
  "node": [
    {
      "floor": int,              // 0…(층수-1)
      "type": int,               // 0:lobby/1:restroom/2:stair/3:elevator/4:office/5:mechanical
      "type_id": int,            // type별 고유 ID
      "center": [z:number,y:number,x:number], // [높이, y, x]
      "region_far": float,       // 해당 노드가 차지하는 면적의 비율 (해당 공간 면적)÷(전체 기준 면적)
      "neighbors":[[floor,type,type_id], …]
    },
    …
  ]
}

※ `edge`는 항상 **양방향**을 표현하세요.
※ `center` 좌표는 **동=+x, 서=–x, 북=+y, 남=–y 남동=+x-y, 북동=+x+y ....** 방향 관계를 수치로 반영해야 합니다.
※ 코어는 일반적으로 다음과 같은 실들의 그룹을 의미합니다 [1:restroom,2:stair,3:elevator,5:mechanical]
※ 1층에는 최소 2개의 로비 노드를 배치하세요

<— 생성 규칙 —>
1. **JSON 형식 엄수**  
   • 출력은 오직 하나의 JSON 객체여야 하며, 추가 텍스트나 주석을 포함하지 않습니다.  
2. **노드 정합성**  
   • 모든 노드에 `floor, region_far, type, type_id, center, neighbors`를 반드시 포함합니다.  
   • `type_id`는 같은 `type` 내에서 0,1,2… 순으로 부여합니다.  
3. **엣지 정합성**  
   • 모든 연결(edge)은 두 방향을 모두 나열하세요.  
   • 사용자가 ‘A와 B가 연결’이라 말하면, A→B, B→A 두 개의 엣지를 각각의 neighbors노드에 반영해야합니다.  
4. **좌표 논리**  
   • “서쪽에 홀이 있다” → x 값은 음수 여유 범위(예: –1.0)  
   • “북동쪽 오피스” → x>0, y>0 값을 부여  
   • 동일층 내 상대적 위치만 정확하면, 절대값은 반드시 √2 같은 복잡한 수가 아니어도 됩니다(1.0, 1.5 등 간단한 실수 허용).  
   • 복도는 한 층에 2개의 0:lobby 노드를 배치하고, 노드를 연결하며 형성합니다.
5. **출력 최소화**  
   • 꼭 필요한 키 외에는 제거하세요.  
   • 공백·줄바꿈은 JSON 포맷에 맞게만 유지(가독성을 위해 들여쓰기 불필요).
6. **Office Variation**  
   • “돌출” 요청이 있는 층의 오피스 노드는, **그 방향**으로 x,y 좌표를 조정하세요(예: 동쪽으로 돌출 -> 해당 노드의 center의 x,y값 +a 하기).  
   • 좌표에 **작은 임의 오프셋**(±3.0) 을 추가해, 완전 동일한 축 정렬을 방지합니다.  
"""

def get_jonning(user_request: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": SYSTEM_1},
            {"role": "user", "content": user_request}
        ],
        temperature=0,
        max_tokens=5000,
        n=1
    )
    return response.choices[0].message.content.strip()

def make_local_graph(user_request: str):
    response = client.chat.completions.create(
        model="ft:gpt-4o-mini-2024-07-18:gyu:local-maker:BRdQiSHT:ckpt-step-131",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_request}
        ],
        temperature=0.01,
        max_tokens=16384,
        n=1
    )
    return response.choices[0].message.content.strip()

def extract_json(text: str):
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    raise ValueError("JSON 본문을 찾을 수 없습니다.")

def save_output_to_json(parsed, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
    print(f"저장 완료: {output_path}")

# ---------------- Streamlit ----------------

with st.spinner("프로그램 조닝을 진행중입니다...⏳"):
    user_prompt = st.session_state.prompt
    width = st.session_state["site_width"]
    depth = st.session_state["site_depth"]
    area = st.session_state["site_area"]
    far = st.session_state["floor_area_ratio"]
    coverage = st.session_state["building_coverage"]
    height_min, height_max = st.session_state["height_limit"]

    # 층수 자동 계산 및 지시문 정제
    floors = int(far // coverage)
    full_prompt = f"{user_prompt}\n{floors}층 오피스 매스를 설계해줘."

    # 1단계: 프로그램 조닝 텍스트
    first_output = get_jonning(full_prompt)
    print("🧾 조닝 결과:\n", first_output)

    # 2단계: 조닝 결과를 바탕으로 local graph 생성
    raw_response = make_local_graph(first_output)

    # 3단계: JSON 저장
    try:
        json_text = extract_json(raw_response)
        local_graph = json.loads(json_text)

        output_path = "Data/6types-raw_data/local_graph_data/local_g.json"
        save_output_to_json(local_graph, output_path)

        st.session_state.graph_data = local_graph
        st.success("✅ 프로그램 조닝 완료!")
    except Exception as e:
        st.error(f"❌ JSON 파싱 실패: {e}")
        st.text_area("📄 원본 응답 내용", raw_response, height=300)