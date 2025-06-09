import os
import torch
from torch_geometric.data import Batch
from Model.models import Generator
from util_eval import save_output, evaluate_best_of_n
from util_graph import get_program_ratio
from train_args import make_args

import os
args = make_args()
if str(args.cuda) == -1:
    cuda = False
else:
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

# GPU 설정하기 쿠다면 쿠다로 설정
device_ids = list(range(torch.cuda.device_count())) if cuda else []
assert(args.batch_size % len(device_ids) == 0)
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if cuda else "Using CPU")
if cuda:
    print([torch.cuda.get_device_name(device_id) for device_id in device_ids])
print(args)
device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')
# device = torch.device("cpu")
print(device)

# 2. 저장된 graph 파일 불러오기 (학습에 사용된 전처리된 데이터)
graph = torch.load("Data/6types-processed_data/data022499.pt", map_location=device)
# 단일 데이터를 배치로 감싸기 (follow_batch 지정)
follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']
graph = Batch.from_data_list([graph], follow_batch=follow_batch).to(device)

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

inference_dir = "inference"
# trained_id = "iccv2021"
# epoch_id = '70'
trained_id = "test"
epoch_id = '20'
trained_file = 'runs/{}/checkpoints/{}.ckpt'.format(trained_id, epoch_id)
# 학습된 체크포인트 파일을 로드 (파일 경로는 실제 체크포인트 경로로 수정)
generator.load_state_dict(torch.load(trained_file), strict=True)
generator.eval()

raw_dir = "Data/6types-raw_data"       # 원본 JSON 파일들이 저장된 폴더 (global_graph_data, local_graph_data, voxel_data 폴더 포함)
output_dir = f"Data/6types-output"        # 결과를 저장할 폴더

import torch

def check_cross_edge_mappings(graph):
    """
    프로그램 노드와 복셀 노드 간의 cross-edge 연결 상태를 확인하는 함수.
    """
    print("📌 프로그램 노드 수:", graph.program_class_feature.shape[0])
    print("📌 복셀 노드 수:", graph.voxel_feature.shape[0])
    
    print("📌 cross_edge_program_index_select:", graph.cross_edge_program_index_select.shape)
    print("📌 cross_edge_voxel_index_select:", graph.cross_edge_voxel_index_select.shape)

    # 연결 비율 계산
    program_indices = graph.cross_edge_program_index_select.cpu().numpy()
    voxel_indices = graph.cross_edge_voxel_index_select.cpu().numpy()
    
    print(f"📌 프로그램-복셀 연결 비율: {len(set(program_indices))} / {graph.program_class_feature.shape[0]}")
    print(f"📌 복셀-프로그램 연결 비율: {len(set(voxel_indices))} / {graph.voxel_feature.shape[0]}")

# 사용 예시
check_cross_edge_mappings(graph)

viz_dir = os.path.join(inference_dir, trained_id, epoch_id + "_", "output")

# 1.0에 가까울수록 더 다양한 결과가 나옴. 
    # 0.7처럼 낮추면 더 일관되고 안정적인 결과가 나옴.
truncated = False
if not truncated:
    trunc_num = 1.0
else:
    trunc_num = 0

# 시각화 출력할 정보를 저장할 디렉토리 생성
graph = graph.to(device)

evaluate_best_of_n(graph, generator, raw_dir, output_dir, follow_batch, device, n_trials=100, trunc=trunc_num)



# program_z = torch.rand([graph.program_class_feature.shape[0], noise_dim]).to(device)
# voxel_z = torch.rand([graph.voxel_feature.shape[0], noise_dim]).to(device)
# out, soft_out, mask, att, max_out_program_index = generator(graph, program_z, voxel_z)
#     # 5. get_program_ratio() 호출하여 추가 정보 계산 (area_index는 학습 시 사용한 index, 예시로 6 사용)
# print(mask["hard"].sum())          # 0이면 모든 voxel이 꺼져 있는 상태
# print(mask["hard"][:10].T)         # 앞쪽 10개 확인

# normalized_program_class_weight, program_weight, FAR = get_program_ratio(graph, att["hard"], mask["hard"], area_index_in_voxel_feature=6)

#     # 6. 저장할 폴더 지정 (raw_dir에는 원본 global, local, voxel JSON 파일들이 있어야 함)
# raw_dir = "Data/6types-raw_data"       # 원본 JSON 파일들이 저장된 폴더 (global_graph_data, local_graph_data, voxel_data 폴더 포함)
# output_dir = f"Data/6types-output"        # 결과를 저장할 폴더
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#     # 하위 폴더들도 없으면 생성 (save_output 내부에서 생성하지만, 미리 확인해도 좋습니다.)
# for sub in ["global_graph_data", "local_graph_data", "voxel_data"]:
#     sub_dir = os.path.join(output_dir, sub)
#     if not os.path.exists(sub_dir):
#             os.makedirs(sub_dir)

#     # 7. 배치 사이즈 (여기서는 단일 데이터이므로 1)
# batch_size = 1

#     # 8. save_output 함수 호출하여 결과 저장
#     # new_data_id_strs는 선택사항입니다.
# batch_all_g = save_output(batch_size, graph, normalized_program_class_weight, program_weight,
#                             FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir)

# print("저장 완료!")
