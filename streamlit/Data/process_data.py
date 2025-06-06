"""
Folder structure:
raw_data
    - global_graph_data
    - local_graph_data
    - voxel_data

"""
import torch
from Data.GraphConstructor import  GraphConstructor
import os
import traceback  # 추가

raw_data_dir = "Data/6types-raw_data"
output_dir = "Data/6types-processed_data"
global_graph_dir = os.path.join(raw_data_dir, "global_graph_data")
local_graph_dir = os.path.join(raw_data_dir, "local_graph_data")
voxel_graph_dir = os.path.join(raw_data_dir, "voxel_data")


for fname in os.listdir(global_graph_dir):
    data_id = int(''.join(filter(str.isdigit, fname)))
    try:
        g = GraphConstructor.load_graph_jsons(data_id, raw_data_dir)
        output_fname = "data" + str(data_id).zfill(GraphConstructor.data_id_length) + ".pt"
        torch.save(g, os.path.join(output_dir, output_fname))
    except Exception as e:
        print(f"Error loading data {str(data_id).zfill(GraphConstructor.data_id_length)}")
        print("Exception:", e)
        traceback.print_exc()  # 상세 오류 출력

print("Data processing completed")

