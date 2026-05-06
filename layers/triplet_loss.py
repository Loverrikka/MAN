import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n


class TripletLoss(nn.Module):

	def __init__(self, margin, normalize_feature=False):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
		assert dist_an.size(0)==dist_ap.size(0)
		y = torch.ones_like(dist_ap)
		loss = self.margin_loss(dist_an, dist_ap, y)
		prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		return loss


class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss


class TripletLossXBM(nn.Module):
    def __init__(self, margin=0.3, norm=False):
        super(TripletLossXBM, self).__init__()
        self.margin = margin
        self.norm = norm
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs_col, targets_col, inputs_row, targets_row):

        n = inputs_col.size(0)
        if self.norm:
            inputs_col = F.normalize(inputs_col)
            inputs_row = F.normalize(inputs_row)

        dist = euclidean_dist(inputs_col, inputs_row)

        # split the positive and negative pairs
        pos_mask = targets_col.expand(
            targets_row.shape[0], n
        ).t() == targets_row.expand(n, targets_row.shape[0])
        neg_mask = ~pos_mask
        # For each anchor, find the hardest positive and negative
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][pos_mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][neg_mask[i]].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss

class MMS_loss(torch.nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss
      
# 定义MMD Loss
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size_source = int(source.size()[0])
        batch_size_target = int(target.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    
    

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, feat3, label1):
        # print("using 3MLoss")
        # print(feat1.shape, feat2.shape, label1.shape, label1)
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        feat3 = feat3.chunk(label_num, 0)
        # loss = Variable(.cuda())
        for i in range(label_num):
          center1 = torch.mean(feat1[i], dim=0)
          center2 = torch.mean(feat2[i], dim=0)
          center3 = torch.mean(feat3[i], dim=0)
          # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
          if self.dist_type == 'l2' or self.dist_type == 'l1':
            if i == 0:
              # print(self.dist(center1, center2), self.dist(center1, center3), self.dist(center2, center3))
              dist = max(abs(self.margin - self.dist(center1, center2)), abs(self.margin - self.dist(center2, center3)), abs(self.margin - self.dist(center1, center3)))
              # dist = max(0, abs(self.margin - self.dist(center1, center2)))
            else:
              dist += max(abs(self.margin - self.dist(center1, center2)), abs(self.margin - self.dist(center2, center3)), abs(self.margin - self.dist(center1, center3)))
              # dist += max(0, abs(self.margin - self.dist(center1, center2)))
        return dist

class Unsupervised_prototype_multiModalMarginLoss(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(Unsupervised_prototype_multiModalMarginLoss, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1_target, feat2_target, feat3_target, 
                      feat1_source, feat2_source, feat3_source):
        # 1. 目标域：计算同一目标域样本的三个模态特征之间的距离，拉近它们
        center1_target = torch.mean(feat1_target, dim=0)
        center2_target = torch.mean(feat2_target, dim=0)
        center3_target = torch.mean(feat3_target, dim=0)
        
        dist_target = max(abs(self.margin - self.dist(center1_target, center2_target)),
                          abs(self.margin - self.dist(center2_target, center3_target)),
                          abs(self.margin - self.dist(center1_target, center3_target)))
        
        # 2. 源域：计算同一源域样本的三个模态特征之间的距离，拉近它们
        center1_source = torch.mean(feat1_source, dim=0)
        center2_source = torch.mean(feat2_source, dim=0)
        center3_source = torch.mean(feat3_source, dim=0)
        
        dist_source = max(abs(self.margin - self.dist(center1_source, center2_source)),
                          abs(self.margin - self.dist(center2_source, center3_source)),
                          abs(self.margin - self.dist(center1_source, center3_source)))

        # 3. 目标域和源域之间：推远目标域与源域样本的模态特征，保持区分性
        dist_source_target = max(abs(self.margin - self.dist(center1_target, center1_source)),
                                 abs(self.margin - self.dist(center2_target, center2_source)),
                                 abs(self.margin - self.dist(center3_target, center3_source)))

        # 总损失：目标域模态对齐 + 源域模态对齐 + 源域与目标域的区分
        total_dist = dist_target + dist_source  + dist_source_target
        return total_dist


class multiModalMarginLossUnsupervised(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossUnsupervised, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)  # Calculate similarity along the feature dimension
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1_target, feat2_target, feat3_target, 
                      prototype1_source, prototype2_source, prototype3_source):

        # Calculate the mean of target features across the batch
        center1_target = torch.mean(feat1_target, dim=0)
        center2_target = torch.mean(feat2_target, dim=0)
        center3_target = torch.mean(feat3_target, dim=0)
        
        # Calculate distances for target domain alignment
        dist_target = max(abs(self.margin - self.dist(center1_target, center2_target)),
                          abs(self.margin - self.dist(center2_target, center3_target)),
                          abs(self.margin - self.dist(center1_target, center3_target)))
        
        # Calculate the mean of source domain prototypes
        center1_source = torch.mean(prototype1_source, dim=0)
        center2_source = torch.mean(prototype2_source, dim=0)
        center3_source = torch.mean(prototype3_source, dim=0)

        # Calculate distances for source domain alignment
        dist_source = max(abs(self.margin - self.dist(center1_source, center2_source)),
                          abs(self.margin - self.dist(center2_source, center3_source)),
                          abs(self.margin - self.dist(center1_source, center3_source)))

        # Calculate distances between source domain prototypes and target domain features
        dist_source_target = max(abs(self.margin - self.dist(center1_target, center1_source)),
                                 abs(self.margin - self.dist(center2_target, center2_source)),
                                 abs(self.margin - self.dist(center3_target, center3_source)))

        # Total loss
        total_dist = dist_target + dist_source + dist_source_target
        return total_dist


class multiModalMarginLossUnsupervised_neo(nn.Module):
    def __init__(self, margin=3, dist_type='l2'):
        super(multiModalMarginLossUnsupervised_neo, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        elif dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)  # 计算特征维度上的余弦相似度
        elif dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1_target, feat2_target, feat3_target, 
                      prototype1_target, prototype2_target, prototype3_target,
                      feat1_source, feat2_source, feat3_source, 
                      prototype1_source, prototype2_source, prototype3_source):

        ### 目标域对齐损失
        # 计算目标域特征的模态中心
        center1_target = torch.mean(feat1_target, dim=0)
        center2_target = torch.mean(feat2_target, dim=0)
        center3_target = torch.mean(feat3_target, dim=0)

        # 计算目标域中每个模态的原型中心
        proto_center1_target = torch.mean(prototype1_target, dim=0)
        proto_center2_target = torch.mean(prototype2_target, dim=0)
        proto_center3_target = torch.mean(prototype3_target, dim=0)

        # 计算目标域中每个模态的原型中心与其他模态的特征中心的对齐损失
        target_loss = (
            abs(self.margin - self.dist(proto_center1_target, center2_target)) + 
            abs(self.margin - self.dist(proto_center1_target, center3_target)) +
            abs(self.margin - self.dist(proto_center2_target, center1_target)) + 
            abs(self.margin - self.dist(proto_center2_target, center3_target)) + 
            abs(self.margin - self.dist(proto_center3_target, center1_target)) + 
            abs(self.margin - self.dist(proto_center3_target, center2_target))
        )

        ### 源域对齐损失
        # 计算源域特征的模态中心
        center1_source = torch.mean(feat1_source, dim=0)
        center2_source = torch.mean(feat2_source, dim=0)
        center3_source = torch.mean(feat3_source, dim=0)

        # 计算源域中每个模态的原型中心
        proto_center1_source = torch.mean(prototype1_source, dim=0)
        proto_center2_source = torch.mean(prototype2_source, dim=0)
        proto_center3_source = torch.mean(prototype3_source, dim=0)

        # 计算源域中每个模态的原型中心与其他模态的特征中心的对齐损失
        source_loss = (
            abs(self.margin - self.dist(proto_center1_source, center2_source)) + 
            abs(self.margin - self.dist(proto_center1_source, center3_source)) +
            abs(self.margin - self.dist(proto_center2_source, center1_source)) + 
            abs(self.margin - self.dist(proto_center2_source, center3_source)) + 
            abs(self.margin - self.dist(proto_center3_source, center1_source)) + 
            abs(self.margin - self.dist(proto_center3_source, center2_source))
        )

        ### 总损失（目标域与源域损失求和）
        total_loss = target_loss + source_loss

        return total_loss


