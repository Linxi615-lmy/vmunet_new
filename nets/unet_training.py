import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.setrecursionlimit(10000)


def _expand_like(weight, ref):
    if weight is None:
        return None
    if weight.dim() == 2:
        weight = weight[:, :, None, None]
    if weight.dim() != 4:
        raise ValueError(f"weight must be 2D or 4D, got shape={tuple(weight.shape)}")
    if weight.shape[0] != ref.shape[0]:
        raise ValueError(f"batch size mismatch: weight {tuple(weight.shape)} vs ref {tuple(ref.shape)}")
    if weight.shape[1] == 1 and ref.shape[1] != 1:
        weight = weight.expand(-1, ref.shape[1], -1, -1)
    elif weight.shape[1] != ref.shape[1]:
        raise ValueError(f"channel mismatch: weight {tuple(weight.shape)} vs ref {tuple(ref.shape)}")
    if weight.shape[2:] != ref.shape[2:]:
        weight = F.interpolate(weight, size=ref.shape[2:], mode="nearest")
    return weight.type_as(ref)

# 计算权重掩码,只支持 none, channel 权重, pixel 权重
def _build_weight_map(ref, valid_mask=None, pixel_weight=None, ifprint=None):
    weight_map = torch.ones_like(ref, dtype=ref.dtype)
    # 得到通道掩码 [B,C,1,1] ,即对那几个标签计算
    if valid_mask is not None:
        if valid_mask.dim() != 2:
            raise ValueError(f"valid_mask must be [B, C], got shape={tuple(valid_mask.shape)}")
        vm = valid_mask[:, :ref.shape[1]].type_as(ref).unsqueeze(-1).unsqueeze(-1)
        weight_map = weight_map * vm
    # print(f"weight_map_valid_mask:{weight_map}")
    if pixel_weight is not None:
        pw = _expand_like(pixel_weight, ref)
        weight_map = weight_map * pw
    if ifprint is not None:
        print(f"weight_map_pixel_weight:{weight_map}")
    return weight_map


def CE_Loss(inputs, target, valid_mask=None, pixel_weight=None, ifprint=None):
    loss_map = F.binary_cross_entropy_with_logits(inputs.float(), target.float(), reduction='none')
    weight_map = _build_weight_map(loss_map, valid_mask=valid_mask, pixel_weight=pixel_weight, ifprint=ifprint)
    denom = weight_map.sum().clamp_min(1.0)
    output = (loss_map * weight_map).sum() / denom
    print(f"CE_Loss_output:{output}")
    return output


def Dice_loss(inputs, targets, valid_mask=None, pixel_weight=None, smooth=1e-5, ifprint=None):
    probs = torch.sigmoid(inputs.float())
    targets = targets.float()
    weight_map = _build_weight_map(probs, valid_mask=valid_mask, pixel_weight=pixel_weight, ifprint=ifprint)

    intersection = (probs * targets * weight_map).sum(dim=(2, 3))
    pred_sum = (probs * weight_map).sum(dim=(2, 3))
    target_sum = (targets * weight_map).sum(dim=(2, 3))

    dice_loss = 1.0 - (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    valid_channel = (weight_map.sum(dim=(2, 3)) > 0).float()
    denom = valid_channel.sum().clamp_min(1.0)
    output = (dice_loss * valid_channel).sum() / denom
    print(f"Dice_Loss_output:{output}")
    return output


def Tversky_loss(inputs, targets, valid_mask=None, pixel_weight=None, alpha=0.3, beta=0.7, smooth=1e-5, ifprint=None):
    probs = torch.sigmoid(inputs.float())
    targets = targets.float()
    weight_map = _build_weight_map(probs, valid_mask=valid_mask, pixel_weight=pixel_weight, ifprint=ifprint)

    tp = (probs * targets * weight_map).sum(dim=(2, 3))
    fp = (probs * (1.0 - targets) * weight_map).sum(dim=(2, 3))
    fn = ((1.0 - probs) * targets * weight_map).sum(dim=(2, 3))

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    loss = 1.0 - tversky
    valid_channel = (weight_map.sum(dim=(2, 3)) > 0).float()
    denom = valid_channel.sum().clamp_min(1.0)
    output = (loss * valid_channel).sum() / denom
    print(f"Tversky_Loss_output:{output}")
    return output


def CeDice_loss(inputs, targets, valid_mask=None, pixel_weight=None, dice_weight=1.0, bce_weight=1.0,ifprint=None):
    ce = CE_Loss(inputs, targets, valid_mask=valid_mask, pixel_weight=pixel_weight,ifprint=ifprint)
    dice = Dice_loss(inputs, targets, valid_mask=valid_mask, pixel_weight=pixel_weight)
    output = bce_weight * ce + dice_weight * dice
    print(f"CeDice_Loss_output:{output}")
    return output


def BCEPlusTversky_loss(inputs, targets, valid_mask=None, pixel_weight=None,
                        bce_weight=1.0, tversky_weight=1.0,
                        tversky_alpha=0.3, tversky_beta=0.7):
    bce = CE_Loss(inputs, targets, valid_mask=valid_mask, pixel_weight=pixel_weight)
    tv = Tversky_loss(
        inputs,
        targets,
        valid_mask=valid_mask,
        pixel_weight=pixel_weight,
        alpha=tversky_alpha,
        beta=tversky_beta,
    )
    return bce_weight * bce + tversky_weight * tv


def Boudaryloss(inputs, targets):
    probs = torch.sigmoid(inputs)
    pc = probs[:, :, ...].type(torch.float32)
    dc = targets[:, :, ...].type(torch.float32)
    multipled = torch.einsum("bkwh,bkwh->bkwh", pc, dc)
    loss = multipled.mean()
    return loss


class focal_loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs, target):
        pred_sigmoid = inputs.sigmoid()
        target = target.type_as(inputs)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none') * focal_weight
        return loss.mean()


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(
                    math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
