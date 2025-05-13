import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...])
        y_sum = torch.sum(target[:, i, ...])
        loss += 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss /= target.shape[1]
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(
        np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, test=None):
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

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        #     # 就应该是两个维度，没有n_views，因此需要修改此处代码。
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)
        if len(features.shape) > 3:
            raise ValueError('`features` needs to be [bsz, features, ...],'
                             '2 dimensions are only required')
            # 就应该是两个维度，没有n_views，因此需要修改此处代码。
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # labels非none
            labels = labels.contiguous().view(-1, 1) # label二维矩阵batch_size*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            # mask是一个batch_size*batch_size的矩阵，每一行代表batch中一个元素的类别情况，其中同类别的其他元素的对应位置为1.0，其余为0
            mask = torch.eq(labels, labels.T).float().to(device) 
        else:
            mask = mask.float().to(device)

        # contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # if self.contrast_mode == 'one':
        #     anchor_feature = features[:, 0]
        #     anchor_count = 1
        # elif self.contrast_mode == 'all':
        #     anchor_feature = contrast_feature
        #     anchor_count = contrast_count
        # else:
        #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        contrast_feature = features
        contrast_count = 1
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # 修改了此处，因为features本来就是两个维度，没有n_views
            anchor_feature = features
            anchor_count = 1
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability，提高数值稳定性
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

        if test is not None:
            print("anchor_feature: ", anchor_feature)
            print("logits: ", logits)
            print("mask: ", mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

def con_loss(client_idx, representations, pro_r, batch_size, temperature=0.5):
    # 通过余弦相似度计算当前batch与其他原型特征之间的contra loss，目的是减少当前batch与其他客户端原型的距离
    # 构建了全0的target向量，目的是让当前batch与其他客户端原型的距离尽可能近
    cos = torch.nn.CosineSimilarity(dim=-1)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    

    # 计算每个 representation 与三个 pro_r 之间的余弦相似度
    cosine_similarities = cos(representations.unsqueeze(1), pro_r[client_idx].unsqueeze(0))
    # 找到每个 representation 与三个 pro_r 的最大余弦相似度
    nega, _ = torch.max(cosine_similarities, dim=1)
    
    # nega = cos(representations, pro_r[client_idx])
    lossCON = 0.0
    for label in pro_r:
        if label != client_idx:
            # min_pro_r_loss = 10000000
            # 计算每个 representation 与三个 pro_r 之间的余弦相似度
            cosine_simi = cos(representations.unsqueeze(1), pro_r[label].unsqueeze(0))
            # 找到每个 representation 与三个 pro_r 的最大余弦相似度
            posi, _ = torch.max(cosine_simi, dim=1)
            # for multi_pro_r in pro_r[label]:
            #     posi = cos(representations, multi_pro_r)
            temp = posi.reshape(-1, 1)
            temp = torch.cat((temp, nega.reshape(-1, 1)), dim=1)
            temp /= temperature
            temp = temp.cuda()
            targets = torch.zeros(batch_size).cuda().long()
            lossCON += criterion(temp, targets)
            #     con = criterion(temp, targets)
            #     if con < min_pro_r_loss:
            #         min_pro_r_loss = con
            # lossCON += min_pro_r_loss
            
    return lossCON * 1.0 / (len(pro_r) - 1)