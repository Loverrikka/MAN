"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# class SupConLoss(nn.Module):
#     def __init__(self, device):
#         super(SupConLoss, self).__init__()
#         self.device = device
#         self.temperature = 1.0
#     def forward(self, text_features, image_features, t_label, i_targets): 
#         batch_size = text_features.shape[0] 
#         batch_size_N = image_features.shape[0] 
#         mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
#             i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

#         logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(logits, dim=1, keepdim=True)
#         logits = logits - logits_max.detach() 
#         exp_logits = torch.exp(logits) 
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
#         loss = - mean_log_prob_pos.mean()

#         return loss
    
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, feat1, feat2, feat3, labels):
        """
        Args:
            feat1, feat2, feat3: 三个模态的特征, 每个的 shape 为 [bsz, dim]
            labels: 样本标签，shape 为 [bsz]，每四个样本共享同一个ID
        Returns:
            一个标量损失值
        """
        device = feat1.device
        batch_size = feat1.shape[0]

        # 对每个模态特征进行L2归一化
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        feat3 = F.normalize(feat3, p=2, dim=1)

        # 拼接特征和标签
        features = torch.cat([feat1, feat2, feat3], dim=0)  # [3 * bsz, dim]
        labels = labels.repeat(3)  # 重复标签以适应拼接后的特征大小

        # 构建正样本对掩码，只在同ID且不同模态之间形成正样本对
        labels = labels.view(-1, 1)
        mask_same_id = torch.eq(labels, labels.T).float().to(device)  # 同ID为1
        mask_diff_modal = torch.block_diag(
            torch.zeros(batch_size, batch_size),
            torch.zeros(batch_size, batch_size),
            torch.zeros(batch_size, batch_size)
        ).to(device)
        mask = mask_same_id * (1 - mask_diff_modal)  # 保留同ID不同模态对

        # 计算相似度
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # 稳定数值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 排除自对比
        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(3 * batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # 计算正样本对的log-likelihood平均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # 计算损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(3, batch_size).mean()

        return loss