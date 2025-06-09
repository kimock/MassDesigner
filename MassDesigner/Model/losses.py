import torch
import torch.nn as nn
import torch_geometric as tg
from util import *
from util_graph import get_program_ratio


import torch
import torch.nn as nn

class VolumetricDesignLoss_D(nn.Module):
    def __init__(self, gan_loss, gp_lambda=10):
        super(VolumetricDesignLoss_D, self).__init__()
        self.gan_loss = gan_loss
        self.gp_lambda = gp_lambda

    def forward(self, real_validity_voxel, fake_validity_voxel, gp=None):
        if self.gan_loss == "WGANGP":
            Dv_loss = -torch.mean(real_validity_voxel[0]) - torch.mean(real_validity_voxel[1]) + torch.mean(fake_validity_voxel[0]) + torch.mean(fake_validity_voxel[1])
            return Dv_loss + (self.gp_lambda * gp if gp is not None else 0)

        device = real_validity_voxel[0].get_device()
        valid0 = torch.FloatTensor(real_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        valid1 = torch.FloatTensor(real_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)
        fake0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(0.0).to(device)
        fake1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(0.0).to(device)

        if self.gan_loss == "NSGAN":  # NS GAN  log(D(x))+log(1-D(G(z)))
            loss = nn.BCELoss()
            return loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1)
        elif self.gan_loss == "LSGAN":  # LS GAN  (D(x)-1)^2 + (D(G(z)))^2
            loss = nn.MSELoss()
            return 0.5 * (loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1))
        elif self.gan_loss == "hinge":  # SA GAN
            loss = nn.ReLU()
            return loss(1.0 - real_validity_voxel[0]).mean() + loss(1.0 - real_validity_voxel[1]).mean() + loss(fake_validity_voxel[0] + 1.0).mean() + loss(fake_validity_voxel[1] + 1.0).mean()
        else:
            raise TypeError("self.gan_loss is not valid")


import torch
import torch.nn as nn
from util import *
from util_graph import get_program_ratio

class VolumetricDesignLoss_G(nn.Module):
    def __init__(self, lp_weight, tr_weight, far_weight, embedding_dim, sample_size, similarity_fun, gan_loss, lp_loss, hinge_margin):
        super(VolumetricDesignLoss_G, self).__init__()
        self.gan_loss = gan_loss
        self.tr_weight = tr_weight
        self.lp_weight = lp_weight
        self.far_weight = far_weight

    def forward(self, fake_validity_voxel, graph, att, mask, area_index_in_voxel_feature):
        device = att.get_device() if att.is_cuda else "cpu"

        # 기존 Loss 구성
        target0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        target1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)

        if self.gan_loss == "WGANGP":
            adversarial_loss = -torch.mean(fake_validity_voxel[0]) - torch.mean(fake_validity_voxel[1])
        elif self.gan_loss == "NSGAN":
            loss = nn.BCELoss()
            adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        elif self.gan_loss == "LSGAN":
            loss = nn.MSELoss()
            adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        elif self.gan_loss == "hinge":
            adversarial_loss = -torch.mean(fake_validity_voxel[0]) - torch.mean(fake_validity_voxel[1])

        # Program Ratio Loss & FAR Loss
        normalized_program_class_weight, _, FAR = get_program_ratio(graph, att, mask, area_index_in_voxel_feature)
        target_ratio_loss = self.tr_weight * nn.functional.smooth_l1_loss(normalized_program_class_weight.flatten(), graph.program_target_ratio)
        far_diff = FAR.view(FAR.shape[0]) - graph.FAR  # FAR 차이 계산
        smooth_loss = nn.functional.smooth_l1_loss(FAR.view(FAR.shape[0]), graph.FAR)  # 기존 FAR 손실


        # **(1) FAR 초과 감지**
        # far_exceed = torch.clamp(far_diff, min=0)  # 초과량만 추출 (음수는 0으로)
        
        # **(2) 초과량에 대해 강한 패널티 적용**
        # far_penalty = (far_exceed * 2).mean()  # 3제곱으로 초과 시 급격히 증가

        # **(3) 적절한 가중치 조절**
        # total_far_loss = self.far_weight * (smooth_loss + far_penalty)
        total_far_loss = self.far_weight * smooth_loss

        # 🔹 Valid Connection Loss 반영 (미분 가능!)
        total_loss = adversarial_loss + target_ratio_loss + total_far_loss

        return total_loss, adversarial_loss, target_ratio_loss, total_far_loss



def compute_gradient_penalty(Dv, batch, label, out):
    # Interpolated sample
    device = out.get_device()
    u = torch.FloatTensor(label.shape[0], 1).uniform_(0, 1).to(device)  # weight between model and gt label
    mixed_sample = torch.autograd.Variable(label * u + out * (1 - u), requires_grad=True).to(device)  # Nv x C
    mask = (mixed_sample.max(dim=-1)[0] != 0).type(torch.float32).view(-1, 1)
    sample = softmax_to_hard(mixed_sample, -1) * mask

    # compute gradient penalty
    dv_loss = Dv(batch, sample)
    grad_b = torch.autograd.grad(outputs=dv_loss[0], inputs=sample, grad_outputs=torch.ones(dv_loss[0].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_s = torch.autograd.grad(outputs=dv_loss[1], inputs=sample, grad_outputs=torch.ones(dv_loss[1].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dv_gp_b = ((grad_b.norm(2, 1) - 1) ** 2).mean()
    dv_gp_s = ((grad_s.norm(2, 1) - 1) ** 2).mean()
    dv_gp = dv_gp_b + dv_gp_s

    return dv_gp, dv_gp_b, dv_gp_s

def compute_vertical_alignment_loss(graph, out_hard, voxel_pos_index=(3,4), stair_label=2, elevator_label=3):
    """
    graph.voxel_feature: (N_voxel, D) e.g. D>=6 (dimension[0..2], coordinate[3..5], etc.)
    out_hard: (N_voxel, num_programs) 1-hot
    voxel_floor_cluster: (N_voxel,) floor index
    stair_label: 2, elevator_label: 3 (사용중인 프로그램 라벨)
    
    - 목표: 
      층 i와 i+1에 대해, stair끼리, elevator끼리 x-y 좌표 차가 작도록
    """
    device = out_hard.device
    # (A) 추출: 복셀별 x, y 좌표
    # 예: voxel_feature에서 x=3, y=4 라고 가정
    x = graph.voxel_feature[:, voxel_pos_index[0]]
    y = graph.voxel_feature[:, voxel_pos_index[1]]

    # (B) 어떤 복셀이 stair/elevator 인지 마스크
    # out_hard[i, stair_label]==1이면 stair
    # out_hard[i, elevator_label]==1이면 elevator
    stair_mask = (out_hard[:, stair_label] == 1)
    elev_mask = (out_hard[:, elevator_label] == 1)

    # (C) floor 정보
    floors = graph.voxel_floor_cluster  # (N_voxel,) = 각 복셀이 몇 번 층인지

    # (D) 수직 정렬 로스 계산
    # 아이디어: 층 i와 i+1의 stair 복셀들 간에 (x_i - x_{i+1})^2 + (y_i - y_{i+1})^2 최소화
    # 간단히 BFS/매칭 등을 할 수도 있지만, 여기서는 예시로 "floor i의 centroid" vs "floor i+1의 centroid" 식으로 접근
    vertical_loss_stair = torch.tensor(0.0, device=device)
    vertical_loss_elev = torch.tensor(0.0, device=device)
    count_stair, count_elev = 0, 0

    max_floor = int(floors.max().item())

    for f in range(max_floor):
        # f층 stair 복셀들의 (x,y) 평균
        f_mask = (floors == f)
        next_mask = (floors == f+1)
        
        # (1) 스테어
        stair_f = f_mask & stair_mask  # 현재 층 stair
        stair_f_next = next_mask & stair_mask  # 다음 층 stair
        if stair_f.sum() > 0 and stair_f_next.sum() > 0:
            x_mean_cur = x[stair_f].mean()
            y_mean_cur = y[stair_f].mean()
            x_mean_next = x[stair_f_next].mean()
            y_mean_next = y[stair_f_next].mean()
            dist = (x_mean_cur - x_mean_next)**2 + (y_mean_cur - y_mean_next)**2
            vertical_loss_stair += dist
            count_stair += 1

        # (2) 엘리베이터
        elev_f = f_mask & elev_mask
        elev_f_next = next_mask & elev_mask
        if elev_f.sum() > 0 and elev_f_next.sum() > 0:
            x_mean_cur = x[elev_f].mean()
            y_mean_cur = y[elev_f].mean()
            x_mean_next = x[elev_f_next].mean()
            y_mean_next = y[elev_f_next].mean()
            dist = (x_mean_cur - x_mean_next)**2 + (y_mean_cur - y_mean_next)**2
            vertical_loss_elev += dist
            count_elev += 1

    # 평균화
    if count_stair > 0:
        vertical_loss_stair /= count_stair
    if count_elev > 0:
        vertical_loss_elev /= count_elev

    vertical_loss = vertical_loss_stair + vertical_loss_elev
    return vertical_loss