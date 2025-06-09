import torch
import os
import json
from Data.GraphConstructor import GraphConstructor
from torch_geometric.data import Batch
from util_graph import get_program_ratio, data_parallel, rebatch_for_multi_gpu
from torch_scatter import scatter_add


def save_output(batch_size, batch, class_weights, program_weights, FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir, new_data_id_strs=None):
    """
    Save the evaluation results
    """
    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))

    cuda = True  # 혹은 args.cuda 값에 따라 결정
    device_ids = list(range(torch.cuda.device_count())) if cuda else []

    num_of_program_node_accum = 0
    batch_all_g = []
    data = rebatch_for_multi_gpu(batch, device_ids, follow_batch, out, class_weights, program_weights, FAR, max_out_program_index)
    """
    --- data ---
    g: graph
    o: voxel label (n[type])
    cw: (n[new_proportion] in global graph) -- program class ratio/weight 
    pw: (n[region_far] in local graph)
    far: (g[far] in global graph)
    pid: the selected program node id for each voxel node
    """


    for i, (g, o, cw, pw, far, pid) in enumerate(data):
        data_id_str = g["data_id_str"][0]
        new_data_id_str = g["data_id_str"][0] if new_data_id_strs is None else str(new_data_id_strs[i]).zfill(GraphConstructor.data_id_length)
        o = o.cpu().data.numpy().tolist()
        cw, pw = cw.cpu().data.numpy(), pw.cpu().data.numpy()
        # 만약 far가 스칼라가 아니라면, 배치 내 i번째 요소를 선택
        if far.ndim == 0:
            far_scalar = far.item()
        else:
            far_scalar = far[i].item()
        # 이후 far_scalar를 사용


        # Modify Global data
        with open(os.path.join(raw_dir, "global_graph_data", "global_g.json")) as f:
            global_graph = json.load(f)

        # far가 텐서라면, 한 원소만 갖는지 확인하고, 여러 원소라면 i번째 값을 선택
        if torch.is_tensor(far):
            far_scalar = far[i].item() if far.numel() > 1 else far.item()
        else:
            far_scalar = far

        global_graph["new_far"] = far_scalar

        for n in global_graph["global_node"]:
            # cw[n['type']]가 텐서면 변환
            proportion = cw[n['type']]
            if torch.is_tensor(proportion):
                proportion = float(proportion.item()) if proportion.dim() == 0 else proportion.tolist()
            n["new_proportion"] = float(proportion)

        with open(os.path.join(output_dir, "global_graph_data", "global_g.json"), 'w') as f:
            json.dump(global_graph, f)



        # Modify Local data
        d = {}  # program id to its type and type id
        with open(os.path.join(raw_dir, "local_graph_data", "local_g.json")) as f:
            local_graph = json.load(f)
        for i, (n, c) in enumerate(zip(local_graph["node"], pw)):
            n["region_far"] = float(c)
            d[i] = [n["type"], n["type_id"]]
        with open(os.path.join(output_dir, "local_graph_data", "local_g.json"), 'w') as f:
            json.dump(local_graph, f)

        # Modify Voxel data
        with open(os.path.join(raw_dir, "voxel_data", "voxel_g.json")) as f:
            voxel_graph = json.load(f)

        label_tensor = out.argmax(dim=1).cpu().numpy()
        row_sum      = out.sum(dim=1).cpu().numpy()  # one-hot 라벨 총합

        for i, n in enumerate(voxel_graph["voxel_node"]):
            if row_sum[i] == 0:
                n["type"] = -1
                n["type_id"] = 0
            else:
                n["type"] = int(label_tensor[i])
                n["type_id"] = 0

        with open(os.path.join(output_dir, "voxel_data", "voxel_g.json"),'w') as f:
            json.dump(voxel_graph, f)

        all_graphs = [global_graph, local_graph, voxel_graph]
        batch_all_g.append(all_graphs)


    return batch_all_g


def evaluate(data_loader, generator, raw_dir, output_dir, follow_batch, device_ids, number_of_batches=0, trunc=1.0):
    number_of_batches = min(number_of_batches, len(data_loader))
    device = device_ids[0]
    with torch.no_grad():
        total_inter, total_program_edge = 0, 0
        for i, g in enumerate(data_loader):
            if i >= number_of_batches:
                break
            program_z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
            program_z = torch.rand(tuple(program_z_shape)).to(device)
            voxel_z_shape = [g.voxel_feature.shape[0], generator.noise_dim]
            voxel_z = torch.rand(tuple(voxel_z_shape)).to(device)
            if trunc < 1.0:
                program_z.clamp_(min=-trunc, max=trunc)
                voxel_z.clamp_(min=-trunc, max=trunc)

            g.to(device)
            out, soft_out, mask, att, max_out_program_index = generator(g, program_z, voxel_z)
            inter_edges, missing_edges, gen_edges = check_connectivity(g, max_out_program_index, mask['hard'])
            total_inter += inter_edges.shape[1]
            total_program_edge += g.program_edge.shape[1]
            normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
            all_g = save_output(data_loader.batch_size, g, normalized_program_class_weight, normalized_program_weight,FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir)

        acc = total_inter/total_program_edge
        print('acc=', acc)
        valid_connection_ratio = total_inter / (total_inter + missing_edges.shape[1] + gen_edges.shape[1])
        print(f"Valid Connection Ratio: {valid_connection_ratio}")

        return all_g


def postprocess_core_labels(out, graph, threshold: float = 0.2, lobby_threshold: float = 0.8):
    CORE_IDS  = torch.tensor([1, 2, 3, 5], device=out.device)
    LOBBY_ID  = 0

    proj       = graph.voxel_projection_cluster.to(out.device)
    floor_idx  = graph.voxel_floor_cluster.to(out.device)
    Nv, n_cls  = out.shape
    num_cols   = int(proj.max().item()) + 1

    ones       = torch.ones(Nv, device=out.device, dtype=torch.long)
    tot_cnt    = scatter_add(ones, proj, dim=0, dim_size=num_cols)
    labels     = out.argmax(dim=1)
    core_mask  = (labels.unsqueeze(1) == CORE_IDS).any(dim=1).long()
    core_cnt   = scatter_add(core_mask, proj, dim=0, dim_size=num_cols)
    core_ratio = core_cnt.float() / tot_cnt.clamp_min(1).float()


    for col_id in range(num_cols):
        if core_cnt[col_id] == 0:
            continue

        vox_idx = (proj == col_id).nonzero(as_tuple=True)[0]
        if vox_idx.numel() == 0:
            continue

        # 이 플래그가 True면 “코어로 통일” 브랜치가 작동한 것
        unified_core = False

        if core_ratio[col_id] >= threshold:
            unified_core = True
            # ---- 코어로 통일 ----
            col_core_labels = labels[vox_idx][(labels[vox_idx].unsqueeze(1) == CORE_IDS).any(dim=1)]
            if col_core_labels.numel():
                counts = torch.tensor([(col_core_labels == cid).sum() for cid in CORE_IDS], device=out.device)
                sel_core_id = CORE_IDS[counts.argmax()].item()
            else:
                sel_core_id = CORE_IDS[0].item()
            out[vox_idx] = 0
            out[vox_idx, sel_core_id] = 1

        else:
            # ---- 코어 제거 및 로비 통일(조건부) ----
            for cid in CORE_IDS:
                out[vox_idx, cid] = 0

            col_labels = labels[vox_idx]
            lobby_cnt = (col_labels == LOBBY_ID).sum()
            non_core_cnt = tot_cnt[col_id] - core_cnt[col_id]
            lobby_ratio = lobby_cnt.float() / non_core_cnt.clamp_min(1).float()
            if lobby_ratio >= lobby_threshold:
                out[vox_idx] = 0
                out[vox_idx, LOBBY_ID] = 1

        # ---- 1층 제외 로비 한 개만 있는 경우 무라벨 처리(코어 통일 시 SKIP) ----
        if not unified_core:
            # 라벨 갱신
            labels = out.argmax(dim=1)
            # 1층이 아닌 voxel
            non_first = vox_idx[floor_idx[vox_idx] != 0]
            # 그 중 로비인 voxel
            lobby_vox = non_first[labels[non_first] == LOBBY_ID]
            if lobby_vox.numel() == 1:
                # out 벡터를 모두 0으로 하면 이후 저장 시 -1 처리
                out[lobby_vox] = 0

    # (2) 11층 초과 voxel 인덱스
    overflow_vox = (floor_idx > 10).nonzero(as_tuple=True)[0]
    for v in overflow_vox:
        col_id    = proj[v].item()
        src_floor = floor_idx[v].item() - 4  # 4층 아래
        # 같은 column(col_id)에서 src_floor에 해당하는 voxel들
        src_idxs = ((proj == col_id) & (floor_idx == src_floor)).nonzero(as_tuple=True)[0]
        if src_idxs.numel() == 0:
            continue
        src = src_idxs[0].item()
        # (3) 만약 source가 오피스(4)였다면,
        if labels[src] == 4:
            # out[v] 벡터를 모두 0으로 초기화한 뒤
            out[v] = 0
            # 오피스 클래스만 1로 설정
            out[v, 4] = 1
        # 아니라면 out[v]는 건드리지 않음

    return out




def evaluate_best_of_n(graph, generator,
                       raw_dir, output_dir, follow_batch,
                       device, n_trials=100, trunc=1.0):
    """
    동일 그래프를 n_trials 번 생성 → VCR 최고(out) 1개만 선택
    선택된 out 에만 postprocess_core_labels() 적용
    """
    best_vcr, best_result, best_meta = -1, None, {}

    g = graph.to(device)          # 그래프는 한 번만 to(device)

    for i in range(n_trials):
        with torch.no_grad():
            # ── 1) 랜덤 noise 생성 ───────────────────────────
            prog_z = torch.rand((g.program_class_feature.size(0),
                                 generator.noise_dim), device=device)
            vox_z  = torch.rand((g.voxel_feature.size(0),
                                 generator.noise_dim), device=device)
            if trunc < 1.0:
                prog_z.clamp_(-trunc, trunc)
                vox_z.clamp_(-trunc, trunc)

            # ── 2) 생성 ────────────────────────────────────
            out, _, mask, att, max_pid = generator(g, prog_z, vox_z)

            # ── 3) 연결성 평가 ─────────────────────────────
            inter, miss, gen = check_connectivity(g, max_pid, mask['hard'])
            vcr = inter.size(1) / (inter.size(1) + miss.size(1) + gen.size(1))
            acc = inter.size(1) / g.program_edge.size(1)

            # ── 4) 베스트 업데이트 ─────────────────────────
            if vcr > best_vcr:
                best_vcr  = vcr
                best_result = (max_pid.clone(), out.clone())   # 저장
                best_meta   = {'vcr': vcr, 'acc': acc,
                               'att': att, 'mask': mask}

    # ─────────────────────────────────────────────────────────────
    # ▶️ 선택된 결과만 후처리
    max_pid, out_best = best_result
    
    out_best = postprocess_core_labels(out_best, g)   # ★ 한 번만!

    # ── 5) 저장 및 리턴 ───────────────────────────────────
    norm_cls_w, norm_prog_w, FAR = get_program_ratio(
        g, best_meta['att']["hard"], best_meta['mask']["hard"], area_index_in_voxel_feature=6)
    


    all_g = save_output(1, g, norm_cls_w, norm_prog_w, FAR,
                        max_pid, out_best, follow_batch,
                        raw_dir, output_dir)

    print(f"\n✅  Best VCR={best_meta['vcr']:.4f}  ACC={best_meta['acc']:.4f}  • saved with post‑processing")
    return all_g



def compute_valid_connection_loss(graph, att_soft, mask_soft):
    """
    valid_connection_loss를 Generator의 Attention Tensor와 직접 연결하여, 학습 가능하게 만드는 함수.
    """
    device = att_soft.device

    # 프로그램-복셀 연결이 유효한지 확인 (soft mask 사용)
    cross_edge_mask = mask_soft[graph.cross_edge_voxel_index_select]  # E x 1 형태
    valid_edges = cross_edge_mask  # soft mask 기반 연결 여부 확인

    # Program-voxel Attention 기반 연결 여부 확인
    program_edges = torch.index_select(att_soft, 0, graph.cross_edge_voxel_index_select)  # E x 1
    valid_program_edges = program_edges * valid_edges  # 올바른 연결만 남김

    # valid_connection_ratio = (유효한 연결 수) / (전체 연결 수)
    valid_connection_ratio = valid_program_edges.sum() / (valid_program_edges.shape[0] + 1e-8)

    # Loss는 높은 valid_connection_ratio를 목표로 함 (미분 가능)
    valid_connection_loss = (1 - valid_connection_ratio)

    return valid_connection_loss



def check_connectivity(g, max_out_program_index, mask):
    """
    Extract connectivity from the generated design
        inter_edges:    program edge observed in the generated output
        missing_edges:  program edges only in the input program graph
        gen_edges:      program edges only in the generated output
    """
    # Look at the g.voxel_edge and see if the two voxel nodes are masked
    voxel_edge_out_mask = mask.reshape([-1])[g.voxel_edge]  # Ev x 2
    sums = torch.sum(voxel_edge_out_mask, dim=0)  # Ev x 1
    masked_edges = g.voxel_edge[:, sums == 2]   # Ev x 2 sums ==2 means voxel edges observed in the generated output

    if masked_edges.shape[1] != 0:
        # Now put program index onto the voxel edge and delete duplicates
        predicted_program_edges = torch.unique(max_out_program_index[masked_edges], dim=1)
        # union of program edges and program edges from the generated output
        mixed_edges = torch.cat((g.program_edge, predicted_program_edges), dim=1)
        unique_mix_edges, mix_counts = mixed_edges.unique(return_counts=True, dim=1)
        inter_edges = unique_mix_edges[:, mix_counts > 1]

        # program edges only in the input program graph
        mixed_gt_edges = torch.cat((g.program_edge, inter_edges), dim=1)
        unique_gt_edges, mix_gt_counts = mixed_gt_edges.unique(return_counts=True, dim=1)
        missing_edges = unique_gt_edges[:, mix_gt_counts == 1]

        # program edges only in the generated output
        mixed_gen_edges = torch.cat((predicted_program_edges, inter_edges), dim=1)
        unique_gen_edges, mix_gen_counts = mixed_gen_edges.unique(return_counts=True, dim=1)
        gen_edges = unique_gen_edges[:, mix_gen_counts == 1]
    else:  # there is no voxel edge
        inter_edges = masked_edges
        missing_edges = g.program_edge
        gen_edges = masked_edges

    return inter_edges, missing_edges, gen_edges


def generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids, trunc=1.0):
    device = device_ids[0]
    batch.to(device)
    with torch.no_grad():

        program_z_shape = [batch.program_class_feature.shape[0], generator.noise_dim]
        program_z = torch.rand(tuple(program_z_shape)).to(device)
        voxel_z_shape = [batch.voxel_feature.shape[0], generator.noise_dim]
        voxel_z = torch.rand(tuple(voxel_z_shape)).to(device)
        if trunc < 1.0:
            program_z.clamp_(min=-trunc, max=trunc)
            voxel_z.clamp_(min=-trunc, max=trunc)

        batch.to(device)
        out, soft_out, mask, att, max_out_program_index = generator(batch, program_z, voxel_z)

        normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(batch, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
        save_output(variation_num, batch, normalized_program_class_weight, normalized_program_weight, FAR, max_out_program_index, out, follow_batch,
                    raw_dir, output_dir, new_data_id_strs=list(range(variation_num)))


def generate_multiple_outputs_from_data(data, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids):
    batch = Batch.from_data_list([data for _ in range(variation_num)], follow_batch)
    generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids)

