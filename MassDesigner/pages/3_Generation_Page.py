import streamlit as st
import json
import os
from vome import *
import math
import sys
from Data.GraphConstructor import *
from make_global import *
import torch
from torch_geometric.data import Batch
from Model.models import Generator
from util_eval import save_output, evaluate_best_of_n
from util_graph import get_program_ratio
from train_args import make_args
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


with st.spinner("매스 디자인을 생성하는 중입니다...⏳"):
    ########################################
    # 데이터 전처리과정
    ########################################

    # 이전 페이지에서 받은 값 확인
    width = st.session_state["site_width"]
    depth = st.session_state["site_depth"]
    area = st.session_state["site_area"]
    far = st.session_state["floor_area_ratio"]
    coverage = st.session_state["building_coverage"]
    height_min, height_max = st.session_state["height_limit"]
    with open("Data/6types-raw_data/local_graph_data/local_g.json", "r") as f:
                uploaded_local_graph = json.load(f)
    # uploaded_local_graph = st.session_state.graph_data

    # 건폐율에 맞춘 건축 가능공간 산정
    base_width = width * math.sqrt(coverage+0.05)
    base_length = depth * math.sqrt(coverage+0.05)


    # 프로그램 그래프를 기반으로 글로벌 그래프 생성
    global_graph = construct_global(uploaded_local_graph, site_area = area, Far = far)
    with open("Data/6types-raw_data/global_graph_data/global_g.json","w") as wf:
        json.dump(global_graph, wf, indent=2)


    # 프로그램 그래프를 기반으로 빈공간 복셀 그래프 생성
    voxel_graph = construct_voxel_stack_line_merge_and_split_keeping_z(
            uploaded_local_graph,
            base_width=base_width,
            base_length=base_length,
            min_gap=2.2,
            max_gap=5,
            use_height_limit=False 
        )
    pre_label_program_nodes(voxel_graph, uploaded_local_graph)

    with open("Data/6types-raw_data/voxel_data/voxel_g.json","w") as wf:
        json.dump(voxel_graph, wf, indent=2)

    # GraphConstructor 활용
    graph = GraphConstructor.load_graph_from_UI(
        global_graph=global_graph,
        local_graph=uploaded_local_graph,
        raw_voxel_graph=voxel_graph,
        data_id=0
    )


    ########################################
    # 추론 밑작업
    ########################################
    args = make_args()
    if str(args.cuda) == -1:
        cuda = False
    else:
        cuda = True
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    # GPU 설정하기 쿠다면 쿠다로 설정
    device_ids = list(range(torch.cuda.device_count())) if cuda else []
    assert(args.batch_size % len(device_ids) == 0)
    device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')
    # device = torch.device("cpu")


    # 2. 노이즈 생성 (학습 시 사용한 값과 동일한 차원)
    noise_dim = args.noise_dim  # 디폴트는 32

    # 3. 학습된 generator 로드
    latent_dim = 128
    generator = Generator(
        program_input_dim=graph.program_class_feature.shape[-1] + 1,  # story level 포함
        voxel_input_dim=graph.voxel_feature.shape[-1],
        hidden_dim=latent_dim,
        noise_dim=noise_dim,
        program_layer=4,
        voxel_layer=12,
        device=device
    ).to(device)

    trained_id = "get_06"
    epoch_id = '100'
    trained_file = 'runs/{}/checkpoints/{}.ckpt'.format(trained_id, epoch_id)
    # 학습된 체크포인트 파일을 로드 (파일 경로는 실제 체크포인트 경로로 수정)
    generator.load_state_dict(torch.load(trained_file), strict=True)
    generator.eval()

    # 1.0에 가까울수록 더 다양한 결과가 나옴. 
        # 0.7처럼 낮추면 더 일관되고 안정적인 결과가 나옴.
    truncated = True
    if not truncated:
        trunc_num = 1.0
    else:
        trunc_num = 0


    # 의미 없지만 함수 그대로 쓰느라 일단 사용
    follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']
    # Batch 객체 생성 시, follow_batch 명시!
    graph = Batch.from_data_list([graph], follow_batch=follow_batch)
    graph = graph.to(device)

    for i in range(3):

        raw_dir = f"Data/6types-raw_data"
        output_dir = f"Data/6types-output{i}"     


        # all_g의 형태: all_graphs = [global_graph, local_graph, voxel_graph]
        all_g = evaluate_best_of_n(graph, generator, raw_dir, output_dir, follow_batch, device, n_trials=150, trunc=trunc_num)
    

    st.success("매스 생성 완료! 다음 단계로 이동합니다.")