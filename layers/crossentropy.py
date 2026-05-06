import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *

class CrossEntropyLabelSmooth(nn.Module):

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

		loss = (- targets * log_probs).mean(0).sum()

		return loss

class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss

class ReverseCrossEntropyLoss(torch.nn.Module):
    def __init__(self, num_classes, epsilon=1e-7):
        """
        初始化反向交叉熵损失函数
        参数:
            num_classes: int - 类别数量
            epsilon: float - 数值稳定项，避免 log(0)
        """
        super(ReverseCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, labels):
        """
        计算反向交叉熵损失
        参数:
            logits: Tensor (B, C) - 模型预测的 logits
            labels: Tensor (B,) - 真实标签
        返回:
            rce_loss: Tensor - 反向交叉熵损失
        """
        # 将 logits 转换为概率分布
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=self.epsilon, max=1.0)

        # 将标签转为 one-hot 编码
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(logits.device)

        # 计算反向交叉熵
        rce_loss = -torch.sum(probs * torch.log(labels_one_hot + self.epsilon), dim=1).mean()
        return rce_loss