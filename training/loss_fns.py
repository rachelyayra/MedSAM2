# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized
import matplotlib.pyplot as plt


def dice_single_loss(inputs, targets, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (inputs * targets).sum(-1)

    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean() 

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects

def center_channel(z):  # shift-invariant to logit offsets => better grads with photometric augs
    return z - z.mean(dim=1, keepdim=True)

def cons_centered_logit_mse(logits_ref, logits_views, mask=None):
    z0 = center_channel(logits_ref).detach()             # stop-grad anchor
    loss, n = 0.0, 0
    for z in logits_views:
        zb = center_channel(z)                           # grads ON
        diff2 = (zb - z0).pow(2)
        if mask is not None:
            if mask.ndim == diff2.ndim - 1: mask = mask.unsqueeze(1)
            diff2 = diff2 * mask
            denom = (mask.sum() * diff2.shape[1]).clamp_min(1.0)
            loss += diff2.sum() / denom
        else:
            loss += diff2.mean()
        n += 1
    return loss / max(n, 1)

def sigmoid_single_focal_loss(
    inputs,
    targets,
    # num_objects,
    step,
    alpha: float = 0.25,
    gamma: float = 2,
    
    loss_on_multimask=False,
    
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    print(f'The shape of inputs in sigmoid function: {inputs.shape, targets.shape}')
    B, C, H, W = inputs.shape
    prob = inputs.sigmoid()

    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    elif targets.dim() != 3:
        raise ValueError(f"targets must be [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}")



    one_hot = F.one_hot(targets.long(), num_classes=C).permute(0,3,1,2).float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot, reduction="none")
    p_t = prob * one_hot + (1 - prob) * (1 - one_hot)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha = [0.60, 0.75, 0.25]
    # print(f'Print: {alpha}')
    if alpha is not None:
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, device=inputs.device, dtype=inputs.dtype)
        elif isinstance(alpha, float):
            alpha = torch.tensor([alpha] * C, device=inputs.device, dtype=inputs.dtype)
        alpha = alpha.view(1, C, 1, 1)  # broadcast to [B, C, H, W]
        alpha_t = alpha * one_hot + (1 - alpha) * (1 - one_hot)
        loss = alpha_t * loss

    return loss.mean()

def sigmoid_single_focal_loss(
    inputs,
    targets,
    # num_objects,
    step = 0,
    alpha: float = 0.25,
    gamma: float = 0,
    
    loss_on_multimask=False,
    
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    print(f'The shape of inputs in sigmoid function: {inputs.shape, targets.shape}')
    B, C, H, W = inputs.shape
    prob = inputs.sigmoid()
    print(f'shape of onehot: {targets.shape}')
    one_hot = F.one_hot(targets.squeeze(1).long(), num_classes=C).permute(0,3,1,2).float()
    for i in range(C):
        save_image(one_hot[0][i].float(), f'gtnew_class_{i}_step_{step}.png')
    ce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot, reduction="none")
    p_t = prob * one_hot + (1 - prob) * (1 - one_hot)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha = [1.0, 1.0, 1.0, 1.0]



    return loss.mean()

def sigmoid_s_t_focal_loss(
    inputs,
    targets,
    # num_objects,
    step = 0,
    alpha: float = 0.25,
    gamma: float = 0,
    
    loss_on_multimask=False,
    
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    print(f'The shape of inputs in sigmoid function: {inputs.shape, targets.shape}')
    B, C, H, W = inputs.shape
    prob = inputs.sigmoid()
    one_hot = targets.sigmoid()
    print(f'shape of onehot: {targets.shape}')
    # one_hot = F.one_hot(targets.squeeze(1).long(), num_classes=C).permute(0,3,1,2).float()
    for i in range(C):
        save_image(one_hot[0][i].float(), f'gtnew_class_{i}_step_{step}.png')
    ce_loss = F.binary_cross_entropy_with_logits(inputs, one_hot, reduction="none")
    p_t = prob * one_hot + (1 - prob) * (1 - one_hot)
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha = [1.0, 1.0, 1.0, 1.0]



    return loss.mean()

def softmax_focal_loss(inputs, targets, alpha = [1.0, 1.0, 1.0, 1.0], gamma = 2.0):
    """
    logits: [B,C,H,W] raw outputs
    targets: [B,H,W] int64 class labels
    alpha: list or tensor of length C (per-class weights)
    """
    # assert inputs.requires_grad, "inputs passed to focal don't  requiregrad"
    # print("[focal] stage0 inputs:", inputs.dtype, inputs.shape, "req", inputs.requires_grad)

    targets = targets.squeeze(1)
    targets = targets.to(torch.long)
    uniq = torch.unique(targets)

    prob = F.softmax(inputs, dim=1).permute(0, 2, 3, 1)  # [N, H, W, C]
    # print(f'The shape of prob: {prob.shape}')
    # Gather probability of the true class at each pixel
    targets_unsqueezed = targets.unsqueeze(-1)  # [N, H, W, 1]
    # print(f'The shape of target: {targets_unsqueezed.shape}')
    p_t = prob.gather(dim=-1, index=targets_unsqueezed).squeeze(-1)  # [N, H, W]

    # Focal modulation
    focal_weight = (1 - p_t) ** gamma

    # Cross-entropy term
    num_classes = 4
    counts = count_classes_from_labels(targets, num_classes)         # targets: [B,H,W] or [B,D,H,W]
    w = weights_effective_number(counts, beta=0.999, clamp_max=5.0)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight = w)  # [N, H, W]
    print("[focal] stage1 ce grad_fn:", type(ce_loss.grad_fn).__name__ if ce_loss.grad_fn else None)

    # Apply modulation
    loss = focal_weight * ce_loss

    # Class weighting if provided
    print(f'Size of targets: {targets.shape} and alpha: {len(alpha)}')
    if alpha is not None:
        if isinstance(alpha, (list, torch.Tensor)):
            # alpha is per-class
            alpha = torch.tensor(alpha, device=inputs.device)
            alpha_t = alpha[targets]   
        else:
            # alpha is scalar
            alpha_t = alpha
        print(f'Size of alpha_t: {alpha_t.shape}')
        loss = alpha_t * loss

    loss = loss.mean()
    print("[focal] stage3 loss_map grad_fn:", type(loss.grad_fn).__name__ if hasattr(loss, 'grad_fn') and loss.grad_fn else None)
    return loss


# def sigmoid_focal_loss(
#     inputs,
#     targets,
#     num_objects,
#     alpha: float = 0.25,
#     gamma: float = 2,
#     loss_on_multimask=False,
# ):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         num_objects: Number of objects in the batch
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         loss_on_multimask: True if multimask prediction is enabled
#     Returns:
#         focal loss tensor
#     """
#     prob = inputs.sigmoid()
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = prob * targets + (1 - prob) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     if loss_on_multimask:
#         # loss is [N, M, H, W] where M corresponds to multiple predicted masks
#         assert loss.dim() == 4
#         return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
#     return loss.mean(1).sum() / num_objects
# def guard_logits_targets(logits, targets, alpha=None, name=""):
#     assert logits.ndim == 4
#     B, C, H, W = logits.shape
#     # collapse one-hot to ids if needed
#     if targets.ndim == 4 and targets.shape[1] > 1:
#         targets = targets.argmax(1)
#     targets = targets.long().contiguous().to(logits.device)
#     tmin, tmax = int(targets.min().item()), int(targets.max().item())
#     uniq = torch.unique(targets).tolist()
#     print(f"[dbg:{name}] logits={tuple(logits.shape)} targets={tuple(targets.shape)} "
#           f"ids∈[{tmin},{tmax}] uniq={uniq[:20]} (C={C}) dev={targets.device}")

#     assert 0 <= tmin, f"targets min {tmin} < 0"
#     assert tmax < C,  f"targets max {tmax} >= C={C}  (remap labels or fix C)"
#     if alpha is not None and isinstance(alpha, (list, tuple, torch.Tensor)):
#         alen = (len(alpha) if not isinstance(alpha, torch.Tensor) else alpha.numel())
#         assert alen == C, f"alpha length {alen} != C={C}"
#     return targets

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def zero_like_with_grad(x: torch.Tensor) -> torch.Tensor:
    # scalar 0.0 tied to graph (keeps requires_grad=True)
    return x.sum() * 0.0

import torch
import torch.nn.functional as F

import math
import torch
import torch.nn.functional as F




def dice_loss_softmax(
    logits: torch.Tensor,           # [B,C,H,W]
    targets: torch.Tensor,          # [B,H,W] or [B,1,H,W]
    *,
    class_weights: torch.Tensor | None = None,  # [C] or [C-1] if exclude_bg
    include_bg: bool = False,
    ignore_index: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)  # [B,C,H,W]

    # normalize targets to [B,H,W]
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets[:, 0]
    elif targets.dim() != 3:
        raise ValueError(f"targets must be [B,H,W] or [B,1,H,W], got {tuple(targets.shape)}")

    # build valid mask for ignore_index
    if ignore_index is not None:
        valid = (targets != ignore_index)
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    else:
        valid = torch.ones_like(targets, dtype=torch.bool)
        safe_targets = targets

    one_hot = F.one_hot(safe_targets.long(), num_classes=C).permute(0,3,1,2).float()
    # zero-out ignored pixels
    one_hot = one_hot * valid.unsqueeze(1)
    probs   = probs   * valid.unsqueeze(1)

    one_hot = one_hot[:, 1:]
    probs   = probs[:, 1:]

    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)           # [C’]
    denom = (probs + one_hot).sum(dims)           # [C’]
    dice  = (2*inter + eps) / (denom + eps)       # [C’]

    # mask out absent classes
    present = (one_hot.sum(dims) > 0).float()     # [C’]
    loss_c = 1.0 - dice                           # [C’]

    if class_weights is not None:
        w = class_weights.to(loss_c.device).float()
        if not include_bg and w.numel() == C:
            w = w[1:]  # drop bg weight to match C'
        if w.numel() != loss_c.numel():
            raise ValueError("class_weights length mismatch.")
        loss = (loss_c * present * w).sum() / (present * w).sum().clamp_min(1.0)
    else:
        loss = (loss_c * present).sum() / present.sum().clamp_min(1.0)

    return loss

def entropy_loss(
    inputs,

    
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    print(f'The shape of inputs in sigmoid function: {inputs.shape}')
    # if step < 5: 
    #     for i in range(len(inputs)):
    #         pred = inputs[i].sigmoid()
    #         for j in range(len(pred)):
    #             save_image(inputs[i][j].float(), f'predraw_step_{step}_{j}.png') 
    #             save_image(pred[j].float(), f'prednew_step_{step}_{j}.png') 
    #             save_image(targets[i][j].float(), f'gtnew_step_{step}_{j}.png') 
    # print(inputs.min(), inputs.max()) 

    B, C, H, W = inputs.shape

    # save_image(inputs.float(), f'predent{0}.png') 
    prob = inputs.sigmoid()
    print(f'print tensor shapes in losses {prob[0].min(), prob[0].max(), inputs[0].shape}')
    img_np = inputs[0].detach().cpu().squeeze().numpy()
    img_np = img_np.transpose(1, 2, 0)
    # plt.imshow(img_np, cmap='gray')  # or cmap='viridis', etc.
    # plt.axis('off')
    # plt.savefig('src_mask0.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    eps = 1e-6
    loss = - (prob * torch.log(prob + eps) + (1 - prob) * torch.log(1 - prob + eps))
    ce_loss = loss.mean()

    return ce_loss

# def softmax_entropy_loss(
#     inputs,

    
# ):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         num_objects: Number of objects in the batch
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#         loss_on_multimask: True if multimask prediction is enabled
#     Returns:
#         focal loss tensor
#     """
#     print(f'The shape of inputs in sigmoid function: {inputs.shape}')
#     # if step < 5: 
#     #     for i in range(len(inputs)):
#     #         pred = inputs[i].sigmoid()
#     #         for j in range(len(pred)):
#     #             save_image(inputs[i][j].float(), f'predraw_step_{step}_{j}.png') 
#     #             save_image(pred[j].float(), f'prednew_step_{step}_{j}.png') 
#     #             save_image(targets[i][j].float(), f'gtnew_step_{step}_{j}.png') 
#     # print(inputs.min(), inputs.max()) 

#     B, C, H, W = inputs.shape

#     # save_image(inputs.float(), f'predent{0}.png') 
#     prob = F.softmax(inputs, dim=1)
#     print(f'print tensor shapes in losses {prob[0].min(), prob[0].max(), inputs[0].shape}')
#     # plt.imshow(img_np, cmap='gray')  # or cmap='viridis', etc.
#     # plt.axis('off')
#     # plt.savefig('src_mask0.png', bbox_inches='tight', pad_inches=0)
#     # plt.close()
#     eps = 1e-6
#     loss = - (prob * torch.log(prob + eps)).sum(dim=1)
#     ce_loss = loss.mean()

#     return ce_loss
    # return loss.mean()


# def softmax_entropy_loss(
#     inputs,                # (B,C,H,W) logits
#     mask=None,             # optional (B,H,W) or (B,1,H,W)
#     mode="hard",           # "hard" | "topk" | "soft"
#     thresh=0.8,            # used if mode="hard"
#     topk_ratio=0.2,        # used if mode="topk"
#     eps=1e-6,
# ):
#     probs = F.softmax(inputs, dim=1)         # (B,C,H,W)
#     conf, _ = probs.max(dim=1)               # (B,H,W)
#     ent = -(probs * (probs.clamp_min(eps)).log()).sum(dim=1)  # (B,H,W)

#     if mask is not None:
#         m = mask if mask.ndim == 3 else mask.squeeze(1)
#         valid = (m > 0).float()
#     else:
#         valid = torch.ones_like(ent)

#     if mode == "hard":
#         keep = ((conf > thresh).float() * valid)
#         loss = (ent * keep).sum() / keep.sum().clamp_min(1.0)

#     elif mode == "topk":
#         conf_masked = conf * (valid > 0)
#         flat = conf_masked.view(-1)
#         k = max(1, int(topk_ratio * flat.numel()))
#         if flat.max() <= 0:  # no valid pixels
#             loss = (ent * valid).sum() / valid.sum().clamp_min(1.0)
#         else:
#             topk_vals, _ = torch.topk(flat, k)
#             cutoff = topk_vals[-1]
#             keep = ((conf_masked >= cutoff).float() * valid)
#             loss = (ent * keep).sum() / keep.sum().clamp_min(1.0)

#     elif mode == "soft":
#         w = conf * valid
#         loss = (ent * w).sum() / w.sum().clamp_min(1.0)

#     else:
#         raise ValueError(f"Unknown mode: {mode}")

#     return loss


# def softmax_entropy_loss(
#     inputs: torch.Tensor,          # logits, shape (B,C,H,W)
#     t_low: float = 0.6,            # gate: keep pixels with pmax in [t_low, t_high]
#     t_high: float = 0.95,
#     tau: float = 1.0,              # optional temperature for entropy computation
#     bg_idx: int | None = None,     # optionally down-weight or exclude background
#     bg_weight: float = 1.0         # 0<bg_weight<=1; 1.0 = no down-weight
# ) -> torch.Tensor:
#     """
#     Confidence-gated softmax entropy for dense predictions.
#     Returns a scalar that requires grad even if no pixels pass the gate.
#     """
#     # ----- compute per-pixel entropy (with grad) -----
#     z = inputs / tau                    # optional temperature to adjust sharpness
#     p = F.softmax(z, dim=1)             # (B,C,H,W)
#     ent_map = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1)  # (B,H,W)

#     # ----- build confidence gate (no grad needed) -----
#     with torch.no_grad():
#         # top-1 prob without materializing full softmax again
#         z_last = inputs.movedim(1, -1)                      # (B,H,W,C)
#         top2, idx = z_last.topk(2, dim=-1)                 # logits: (...,2)
#         top1_logit = top2[..., 0]
#         lse = torch.logsumexp(z_last, dim=-1)              # (B,H,W)
#         pmax = (top1_logit - lse).exp()                    # (B,H,W)
#         gate = (pmax >= t_low) & (pmax <= t_high)          # boolean mask

#         # optional: background down-weight (do NOT exclude by default)
#         if bg_idx is not None and bg_weight < 1.0:
#             yhat = idx[..., 0]                             # argmax class (B,H,W)
#             gate = gate.float()
#             gate[yhat == bg_idx] *= bg_weight              # reduce bg influence
#         else:
#             gate = gate.float()

#     # ----- weighted mean over selected pixels -----
#     num = (ent_map * gate).sum()
#     den = gate.sum()
#     loss = num / den.clamp_min(1e-8) if den > 0 else inputs.sum() * 0.0  # graph-tied zero
#     return loss


def softmax_entropy_loss(
    inputs,                      # (B,C,H,W) logits
    t_low=0.6, t_high=0.95,      # confidence band on pmax
    p2_min=0.15,                 # require overlap: p2 >= p2_min
    tau=1.4,                     # soften for more signal
    bg_idx=None, bg_weight=1.0,  # optional: down-weight background
    boundary_boost=0.0           # e.g., 1.0 -> 2x weight on edges
):
    # entropy (with grad)
    z = inputs / tau
    p = F.softmax(z, dim=1)                    # (B,C,H,W)
    ent_map = -(p * (p.clamp_min(1e-8)).log()).sum(dim=1)  # (B,H,W)

    # gating (no grad)
    with torch.no_grad():
        z_last = inputs.movedim(1, -1)         # (B,H,W,C)
        top2, idx = z_last.topk(2, dim=-1)     # logits: (...,2)
        top1_logit, second_logit = top2[...,0], top2[...,1]
        lse = torch.logsumexp(z_last, dim=-1)  # (B,H,W)
        pmax = (top1_logit - lse).exp()
        p2   = (second_logit - lse).exp()
        gate = (pmax >= t_low) & (pmax <= t_high) & (p2 >= p2_min)

        w = gate.float()
        if bg_idx is not None and bg_weight < 1.0:
            yhat = idx[...,0]
            w[yhat == bg_idx] *= bg_weight

        if boundary_boost > 0:
            # simple argmax-edge detector
            y = z.argmax(dim=-1)  # (B,H,W)
            edge = (y[:,1:,:] != y[:,:-1,:]) | (y[:,:,1:] != y[:,:, :-1])
            edge = torch.nn.functional.pad(edge.float(), (0,1,0,1))
            w *= (1.0 + boundary_boost * edge)

    num = (ent_map * w).sum()
    den = w.sum()
    return num / den.clamp_min(1e-8) if den > 0 else inputs.sum() * 0.0

def softmax_entropy_loss(
    inputs,                      # (B,C,H,W) logits
):
    # entropy (with grad)
    z = inputs.float()
    if not torch.isfinite(z).all():
        raise ValueError("Logits contain NaN/Inf")
    p = F.softmax(z, dim=1)                    # (B,C,H,W)

    with torch.no_grad():
        assert torch.isfinite(p).all(), "p has NaN/Inf"
        print("p range:", float(p.min()), float(p.max()))
        sums = p.sum(dim=1)                        # (B,H,W)
        print("sum over classes ~1:", float(sums.mean()))
        print("max prob mean:", float(p.max(dim=1).values.mean()))
    log_p = F.log_softmax(z, dim=1) 
    
    ent_map = -(p * (log_p)).sum(dim=1)  # (B,H,W)
    loss = ent_map.mean()
    return loss , sums


def margin_fg_vs_bg_loss(logits, alpha=10.0):
    # union-foreground logit via logsumexp for smooth OR
    fg = torch.logsumexp(logits[:, 1:], dim=1)   # (B,H,W)
    bg = logits[:, 0]                            # (B,H,W)
    margin = fg - bg                             # >0 => fg preferred
    # maximize |margin| to reduce ambiguity (bg vs fg)
    # softplus(-|m|) penalizes small margins without exploding grads
    return F.softplus(-alpha * margin.abs()).mean()

@torch.no_grad()
def _top1_top2(z_last):
    # z_last: (..., C)
    top2, idx = z_last.topk(2, dim=-1)
    top1, top2log = top2[..., 0], top2[..., 1]
    yhat = idx[..., 0]
    lse  = torch.logsumexp(z_last, dim=-1)
    p1   = (top1 - lse).exp()       # top-1 prob
    p2   = (top2log - lse).exp()    # second prob
    return p1, p2, yhat

def non_overlap_gap_loss(
    logits: torch.Tensor,          # (B,C,H,W)
    *,
    bg_idx: int = 0,
    t_low: float = 0.6,            # gate by p1 lower bound
    t_high: float = 0.95,          # gate by p1 upper bound
    p2_min: float = 0.15,          # require some runner-up prob
    gamma: float = 0.4,            # desired margin (logit units)
    alpha: float = 10.0,           # smooth hinge sharpness
    fg_weight: float = 1.0,
    bg_weight: float = 0.25,
) -> torch.Tensor:
    # Move class axis last for convenience
    z = logits.movedim(1, -1)                  # (B,H,W,C)

    # -------- gating (no grad): keep ambiguous pixels only --------
    with torch.no_grad():
        # assumes _top1_top2 returns p1, p2, yhat from logits z
        p1, p2, yhat = _top1_top2(z)           # implement or replace if needed
        band    = (p1 >= t_low) & (p1 <= t_high)
        overlap = (p2 >= p2_min)
        w = (band & overlap).float()           # (B,H,W)

        # optionally down-weight pixels predicted as background
        w = w * torch.where(
            (yhat == bg_idx),
            torch.full_like(w, bg_weight),
            torch.full_like(w, fg_weight),
        )

    # -------- margin with grad: top-1 vs top-2 logit gap --------
    top2_vals, _ = z.topk(2, dim=-1)           # (...,2)
    gap = top2_vals[..., 0] - top2_vals[..., 1]
    lossmap = F.softplus(alpha * (gamma - gap)) / alpha

    num = (lossmap * w).sum()
    den = w.sum()
    if den.item() == 0:
        return logits.sum() * 0.0              # zero with grad path
    return num / den

def original_non_overlap_gap_loss(
    logits,                      # (B,C,H,W)
    *,
    bg_idx: int = 0,
    t_low: float = 0.6,           # p1 lower gate
    t_high: float = 0.95,         # p1 upper gate
    p2_min: float = 0.15,         # require some overlap
    gamma: float = 0.4,
    alpha: float = 10.0,
    bg_weight: float = 0.25,
    tau: float = 1.3
):
    # Move class axis last for convenience
    
    z = logits.movedim(1, -1)                 # (B,H,W,C)

    with torch.no_grad():
        p1, p2, yhat = _top1_top2(z)
        band   = (p1 >= t_low) & (p1 <= t_high)
        overlap= (p2 >= p2_min)
        w = (band & overlap).float()
        if bg_idx is not None:
            w_bg = (yhat == bg_idx) & band & overlap
            w[w_bg] *= bg_weight

    # Compute margin with grad
    z_soft = z 
    top2_vals, _ = z_soft.topk(2, dim=-1)
    gap = top2_vals[..., 0] - top2_vals[..., 1]
    lossmap = F.softplus(alpha * (gamma - gap)) / alpha

    num = (lossmap * w).sum()
    den = w.sum()
    return num / den.clamp_min(1e-8) if den > 0 else logits.sum() * 0.0

def consistency_loss(
    logits_list,        # [ref_logits, aug1_logits, aug2_logits, ...]
    temp: float = 1.0,  # use 0.7 to sharpen if you like
    mask: torch.Tensor = None  # [N,1,H,W(,T)] or [N,H,W(,T)] or None
):
    """
    Multilabel consistency: average MSE between sigmoid(ref) and sigmoid(aug_i).
    - Anchor (ref) is detached so gradients flow only into augmented predictions.
    - Returns the mean over elements and over augmented views.
    - Zero when aug predictions equal anchor predictions (even if soft).
    """
    assert len(logits_list) >= 2, "Need reference + at least one augmented view"

    # 1) anchor probs (stop-grad)
    ref_logits = logits_list[0].detach()
    p_ref = torch.sigmoid(ref_logits / temp)  # [N,C,...]

    # Optional confidence mask (keep pixels with any confident class)
    if mask is None:
        # per-pixel confidence via max across classes
        conf = p_ref.max(dim=1).values  # [N,...]
        mask = (conf >= 0.7).float()    # tune threshold
    if mask.ndim == p_ref.ndim - 1:
        mask = mask.unsqueeze(1)        # [N,1,...] to broadcast over C

    # 2) compute averaged MSE across augmented views
    loss = 0.0
    count = 0
    for z in logits_list[1:]:
        p = torch.sigmoid(z / temp)     # grads ON
        diff = (p - p_ref) * mask       # [N,C,...]
        # mean over all active elements
        active = (mask.sum() * p.shape[1]).clamp_min(1.0)
        loss += (diff.pow(2).sum() / active)
        count += 1

    return loss / max(count, 1)

def _center_logits(z):           # [N,C,(T),H,W]
    return z - z.mean(dim=1, keepdim=True)

def consistency_bce_multilabel(logits_list, mask=None, tau=0.7):
    """
    Multilabel consistency with BCE-with-logits against a detached soft target.
    - logits_list: [ref_logits, aug1_logits, aug2_logits, ...], shape [N,C,(T,)H,W]
    - mask: optional [N,1,(T,)H,W] or [N,(T,)H,W]; if None, build from anchor confidence
    Returns: mean over elements and views (scale ~ O(1)).
    """
    assert len(logits_list) >= 2
    ref_logits = logits_list[0].detach()

    # soft target in [0,1] (NO grad)
    with torch.no_grad():
        p_ref = torch.sigmoid(ref_logits)  # [N,C,...]
        if mask is None:
            conf = p_ref.max(dim=1).values             # [N,...]
            mask = (conf >= tau).float()
        if mask.ndim == p_ref.ndim - 1:
            mask = mask.unsqueeze(1)                   # [N,1,...]

    loss, n = 0.0, 0
    for z in logits_list[1:]:
        per_elem = F.binary_cross_entropy_with_logits(z, p_ref, reduction='none')  # [N,C,...]
        per_elem = per_elem * mask
        denom = (mask.sum() * z.shape[1]).clamp_min(1.0)  # normalize by active elements × C
        loss += per_elem.sum() / denom
        n += 1
    return loss / max(n, 1)

def cons_ce_simple(student_logits, teacher_logits_or_probs, mask=None, eps=1e-3):
    
    # CE(t || s): student compared to smoothed teacher probs
    with torch.cuda.amp.autocast(enabled=False):
        # (a) student: NO extra sharpening → T = 1.0
        s_logp = F.log_softmax(student_logits.float(), dim=1)

        # (b) teacher: make probs, normalize, then tiny label smoothing
        t = teacher_logits_or_probs.float()
        if t.min() < 0 or t.max() > 1:          # looks like logits → softmax
            t = F.softmax(t, dim=1)
        t = t.clamp_min(1e-12)
        t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-12)
        t = (1.0 - eps) * t + eps / t.size(1)   # smooth
        t = t.detach()                          # no grad to teacher

        per_pix = -(t * s_logp).sum(dim=1)      # [B,H,W]
        if mask is not None:
            m = mask if mask.ndim == per_pix.ndim else mask.squeeze(1)
            m = m.float()
            return (per_pix * m).sum() / m.sum().clamp_min(1.0)
        return per_pix.mean()

def cons_ce_multi(students, teacher_logits_or_probs, masks=None, eps=1e-3):
    losses = []
    for i, s in enumerate(students):
        m = None if masks is None else masks[i]
        losses.append(sigmoid_s_t_focal_loss(s, teacher_logits_or_probs) + dice_single_loss(s, teacher_logits_or_probs))
    return torch.stack(losses).mean() 

def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def compactness_loss(
    embeds, logits,
    conf_thresh=0.85, min_pixels=64,
    background_class=0, bg_weight=0.25,
    resize_mode="bilinear"
):

    device = embeds.device
    B, C, H_e, W_e = embeds.shape
    K = logits.shape[1]
    print(f'compactness loss shapes: embeds {embeds.shape}, logits {logits.shape}')

    # downscale 
    probs = F.softmax(logits, dim=1)
    if (logits.shape[-2], logits.shape[-1]) != (H_e, W_e):
        if resize_mode in {"bilinear","bicubic","trilinear","linear"}:
            probs = F.interpolate(probs, size=(H_e, W_e), mode=resize_mode, align_corners=False)
        else:
            probs = F.interpolate(probs, size=(H_e, W_e), mode=resize_mode)  

    
    P = probs.permute(0,2,3,1).reshape(-1, K)   # (N,K)

    print(f'compactness loss shapes after interp: probs {probs.shape}, P {P.shape}')
    conf, pred = P.max(dim=1)                   # (N,), (N,)
    print(f'compactness loss shapes after max: conf {conf.shape}, pred {pred.shape}'    )
    z = embeds.view(32,-1)                         # (N,)
    print(f'compactness loss shapes after view: z {z.shape}'    )

    # per-class variance
    per_vals, per_w = [], []
    for k in range(K):
        m = (pred == k) & (conf > conf_thresh)
        n = int(m.sum())
        if n >= min_pixels:
            zk = z[:, m]
            mu = zk.mean()
            var = (zk - mu).pow(2).mean()
            w = (bg_weight if k == background_class else 1.0)
            per_vals.append(var * w); per_w.append(w)

    if not per_vals:
        return torch.tensor(0.0, device=device)

    vals = torch.stack(per_vals)
    w = torch.tensor(per_w, device=device, dtype=vals.dtype)
    w = w / (w.sum() + 1e-8)
    return (vals * w).sum()

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        target_masks = targets.unsqueeze(1).float()
        # print(f'The shape of inputs in forward the second: {target_masks.shape}')
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'Losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        
        if loss_multimask.size(1) > 1:
            # Find the one with the least 
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # backprop focal, dice and iou loss only if obj present
        # print(f'Target class and objects{target_obj.shape, target_obj}')
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # sum over batch dimension (note that the losses are already divided by num_objects)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

class MultiStepSingleTTAMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')
        # # print(f'The shapes of stuff {len(outs_batch), targets_batch.shape}')
        # self.step_count = 0
        # num_objects = torch.tensor(
        #     (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        # )  # Number of objects is fixed within a batch
        # # print(f'The number of objects {num_objects}')
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_objects)
        # num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs in outs_batch:
            # For each frame
            print(f'the size of the target {len(outs)}')
            cur_losses = self._forward(outs)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses

    def _forward(self, outputs: Dict):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        # target_masks = targets.unsqueeze(1).float()
        # target_masks = targets.float()
        # Get the multistep pred for each frame
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        # print(f'The shape of inputs in the srcs: {len(src_masks_list), src_masks_list[0].shape}')
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # print('This is the second loop')
            self._update_losses(
                losses, src_masks
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        
        return losses

    def _update_losses(
        self, losses, src_masks
    ):
        src_masks = src_masks.squeeze(0)
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_masks.shape) == 3:
            src_masks = src_masks.unsqueeze(0)
            # target_masks = target_masks.unsqueeze(0)
        # target_masks = target_masks.view(-1, 3, 512, 512) 
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = entropy_loss(
            src_masks,
        )
        # loss_multidice = dice_single_loss(
        #     src_masks, target_masks
        # )

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        # loss_dice = loss_multidice
        
        losses["loss_mask"] += loss_mask.sum()
        # print(f'The losses: {loss_mask.sum()}')
        losses["loss_dice"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss
    
class MultiStepSingleTTAConMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            cur_losses = self._forward(frame_input)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses


    def _forward(self, outputs: List[Dict]):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask = entropy_loss(
            src_mask,
        )
        # loss_multidice = consistency_loss(src_masks)
        con_masks = src_masks[1:]
        loss_multidice = cons_ce_multi(con_masks[1:], con_masks[0])

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss
    
class MultiStepSingleTTAMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')
        # # print(f'The shapes of stuff {len(outs_batch), targets_batch.shape}')
        # self.step_count = 0
        # num_objects = torch.tensor(
        #     (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        # )  # Number of objects is fixed within a batch
        # # print(f'The number of objects {num_objects}')
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_objects)
        # num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs in outs_batch:
            # For each frame
            print(f'the size of the target {len(outs)}')
            cur_losses = self._forward(outs)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses

    def _forward(self, outputs: Dict):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        # target_masks = targets.unsqueeze(1).float()
        # target_masks = targets.float()
        # Get the multistep pred for each frame
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        # print(f'The shape of inputs in the srcs: {len(src_masks_list), src_masks_list[0].shape}')
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # print('This is the second loop')
            self._update_losses(
                losses, src_masks
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        
        return losses

    def _update_losses(
        self, losses, src_masks
    ):
        src_masks = src_masks.squeeze(0)
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_masks.shape) == 3:
            src_masks = src_masks.unsqueeze(0)
            # target_masks = target_masks.unsqueeze(0)
        # target_masks = target_masks.view(-1, 3, 512, 512) 
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = entropy_loss(
            src_masks,
        )
        # loss_multidice = dice_single_loss(
        #     src_masks, target_masks
        # )

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        # loss_dice = loss_multidice
        
        losses["loss_mask"] += loss_mask.sum()
        # print(f'The losses: {loss_mask.sum()}')
        losses["loss_dice"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss
    
class MultiStepSingleTTAConMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            cur_losses = self._forward(frame_input)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses


    def _forward(self, outputs: List[Dict]):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask = entropy_loss(
            src_mask,
        )
        # loss_multidice = consistency_loss(src_masks)
        con_masks = src_masks[1:]
        loss_multidice = cons_cosine_logit(con_masks[0], con_masks[1:])

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss

class MultiStepSingleTTAMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')
        # # print(f'The shapes of stuff {len(outs_batch), targets_batch.shape}')
        # self.step_count = 0
        # num_objects = torch.tensor(
        #     (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        # )  # Number of objects is fixed within a batch
        # # print(f'The number of objects {num_objects}')
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_objects)
        # num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs in outs_batch:
            # For each frame
            print(f'the size of the target {len(outs)}')
            cur_losses = self._forward(outs)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses

    def _forward(self, outputs: Dict):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        # target_masks = targets.unsqueeze(1).float()
        # target_masks = targets.float()
        # Get the multistep pred for each frame
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        # print(f'The shape of inputs in the srcs: {len(src_masks_list), src_masks_list[0].shape}')
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # print('This is the second loop')
            self._update_losses(
                losses, src_masks
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        
        return losses

    def _update_losses(
        self, losses, src_masks
    ):
        src_masks = src_masks.squeeze(0)
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_masks.shape) == 3:
            src_masks = src_masks.unsqueeze(0)
            # target_masks = target_masks.unsqueeze(0)
        # target_masks = target_masks.view(-1, 3, 512, 512) 
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        loss_multimask = softmax_entropy_loss(
            src_masks,
        )
        # loss_multidice = dice_single_loss(
        #     src_masks, target_masks
        # )

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        # loss_dice = loss_multidice
        
        losses["loss_mask"] += loss_mask.sum()
        # print(f'The losses: {loss_mask.sum()}')
        losses["loss_dice"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss
    
class MultiStepSoftMaxConMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            cur_losses = self._forward(frame_input)
            for k, v in cur_losses.items():
                losses[k] += v


        return losses


    def _forward(self, outputs: List[Dict]):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]
        all_upscaled_embeddings = [output["multistep_upscaled_embeddings"] for output in outputs]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps, all_upscaled_embeddings[step_idx]
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks, upscaled_embeddings
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        upscaled_embedding = upscaled_embeddings[0]
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask = softmax_entropy_loss(
            src_mask
        )
        # loss_iou = original_non_overlap_gap_loss(src_mask)
        # loss_multidice = consistency_loss(src_masks)
        con_masks = src_masks[1:]
        loss_multidice = cons_ce_multi(con_masks[1:], con_masks[0])
        # loss_class = compactness_loss(upscaled_embedding, src_mask)

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss

class MultiStepSoftMaxTTAConMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict]):
        print(f'The shapes of the inputs {len(outs_batch)}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            cur_losses = self._forward(frame_input)
            for k, v in cur_losses.items():
                losses[k] += v


        return losses


    def _forward(self, outputs: List[Dict]):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]
        all_upscaled_embeddings = [output["multistep_upscaled_embeddings"] for output in outputs]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps, all_upscaled_embeddings[step_idx]
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks, upscaled_embeddings
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        upscaled_embedding = upscaled_embeddings[0]
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask = softmax_entropy_loss(
            src_mask
        )
        # loss_iou = original_non_overlap_gap_loss(src_mask)
        # loss_multidice = consistency_loss(src_masks)
        con_masks = src_masks[1:]
        # loss_multidice = cons_ce_multi(con_masks[1:], con_masks[0])
        # loss_class = compactness_loss(upscaled_embedding, src_mask)

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_multidice = softmax_focal_loss(con_masks[1:], target)
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss



class MultiStepSoftMaxSemiSupConMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict], target=None):
        # print(f'The shapes of the inputs {target.shape}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            print(f'Length of inputs: {len(frame_input)}')
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            if target is not None:
                cur_losses = self._forward(frame_input, target[frame_idx])
            else:
                cur_losses = self._forward(frame_input)
            for k, v in cur_losses.items():
                losses[k] += v


        return losses


    def _forward(self, outputs: List[Dict], target=None):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]
        all_upscaled_embeddings = [output["multistep_upscaled_embeddings"] for output in outputs]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'The length of con_masks: {len(con_masks_steps)}')
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps, all_upscaled_embeddings[step_idx], target
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks, upscaled_embeddings, target=None
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        upscaled_embedding = upscaled_embeddings[0]
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask, p = softmax_entropy_loss(
            src_mask
        )
        # loss_iou = original_non_overlap_gap_loss(src_mask)
        # loss_multidice = consistency_loss(src_masks)
        sup_masks = src_masks[1]
        # print(len(f'Length of Con masks{len(sup_masks)}'))
        # loss_multidice = cons_ce_multi(con_masks[1:], con_masks[0])
        # loss_class = compactness_loss(upscaled_embedding, src_mask)

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        teacher_masks = src_masks[2]
        con_masks = src_masks[3:]
        loss_multidice =  dice_loss_softmax(sup_masks, target) +   (0.5 * boundary_loss_from_labels(sup_masks, target, (1.0, 1.0 )))
        

        loss_mask = loss_multimask
        loss_dice = loss_multidice
        # loss_class = torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        loss_iou = cons_ce_multi(con_masks, teacher_masks)

        # print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss
    
class MultiStepSingleMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        print(f'The shapes of the inputs in first forward {len(outs_batch), targets_batch.shape}')
        # The shapes of the inputs in first forward (2, torch.Size([2, 12, 512, 512]))

        self.step_count = 0

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            # For each frame
            print(f'the size of the target {len(outs)}')
            # the size of the target 13
            cur_losses = self._forward(outs, targets)
            for k, v in cur_losses.items():
                losses[k] += v

        # print(f'This is the losses{losses.items()}')
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor):
  
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        # target_masks = targets.unsqueeze(1).float()
        target_masks = targets.float()
        # Get the multistep pred for each frame
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]
        # print(f'The shape of inputs in the srcs: {len(src_masks_list), src_masks_list[0].shape}')
        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            # print('This is the second loop')
            print(f'The shapes of the inputs in first forward per step {src_masks.shape, target_masks.shape}')
            # The shapes of the inputs in first forward per step (torch.Size([4, 3, 512, 512]), torch.Size([12, 512, 512]))
            self._update_losses(
                losses, src_masks, target_masks, ious,object_score_logits
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, object_score_logits
    ):
        src_masks = src_masks.squeeze(0)
        
        if len(src_masks.shape) == 3:
            src_masks = src_masks.unsqueeze(0)
            target_masks = target_masks.unsqueeze(0)
        target_masks = target_masks.view(-1, 3, 512, 512) 
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        # The shape of inputs before entry to loss calcs: (torch.Size([4, 3, 512, 512]), torch.Size([4, 3, 512, 512]))

        loss_multimask = sigmoid_single_focal_loss(
            src_masks,
            target_masks,
            # num_objects,
            self.step_count,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_single_loss(
            src_masks, target_masks, loss_on_multimask=True
        )

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        loss_mask = loss_multimask
        loss_dice = loss_multidice
        
        losses["loss_mask"] += loss_mask.sum()
        # print(f'The losses: {loss_mask.sum()}')
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        # print(f'Print the reduced: {reduced_loss}')
        return reduced_loss


class MultiStepMultiSoftMaxMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        target_masks = targets.unsqueeze(1).float()
        # print(f'The shape of inputs in forward the second: {target_masks.shape}')
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        loop_count = 0

        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):  
            print(f'This is the loop count: {loop_count, src_masks.shape}')

            loop_count += 1
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        # guard_logits_targets(src_masks, target_masks)

        loss_multimask = softmax_focal_loss(
            src_masks,
            target_masks,
            # num_objects,
            # alpha=self.focal_alpha,
            # gamma=self.focal_gamma,
            # loss_on_multimask=True,
        )
        # print(f"Requires Grad{src_masks.requires_grad}")   # should be True
        print(src_masks.grad_fn) 


        loss_multidice = dice_loss_softmax(
            src_masks, target_masks
        )

        # l1 = margin_fg_vs_bg_loss(
        #     src_masks,
        # )

        l2 = original_non_overlap_gap_loss(src_masks)

        loss_iou =  l2
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # loss_multiiou = iou_loss(
        #     src_masks,
        #     target_masks,
        #     ious,
        #     # num_objects,
        #     loss_on_multimask=True,
        #     use_l1_loss=self.iou_use_l1_loss,
        # )

        print(f'The multimask " {loss_multimask.sum()}')


        loss_mask = loss_multimask
        loss_dice = loss_multidice
        # loss_iou = loss_multiiou

        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class.sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss

class MultiStepMultiSigmoidMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # print(f'The shape of inputs in forward the first: {targets.shape}')
        target_masks = targets.unsqueeze(1).float()
        # print(f'The shape of inputs in forward the second: {target_masks.shape}')
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        loop_count = 0

        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):  
            print(f'This is the loop count: {loop_count, src_masks.shape}')

            loop_count += 1
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        # target_masks = target_masks.expand_as(src_masks)
        # get focal, dice and iou loss on all output masks in a prediction step
        # guard_logits_targets(src_masks, target_masks)

        loss_multimask = sigmoid_single_focal_loss(
            src_masks,
            target_masks,
            # num_objects,
            # alpha=self.focal_alpha,
            # gamma=self.focal_gamma,
            # loss_on_multimask=True,
        )
        # print(f"Requires Grad{src_masks.requires_grad}")   # should be True
        print(src_masks.grad_fn) 

        # logits_dbg = src_masks.detach().clone().requires_grad_(True)
        # L = softmax_focal_loss(logits_dbg, target_masks)   # call exactly as in training
        # print(f'L: {L}m {logits_dbg.shape}')
        # g = torch.autograd.grad(loss_multimask, src_masks, retain_graph=True)[0]
        # print("||dL/dlogits||_mean:", g.abs().mean().item(), "  shape:", g.shape)
        # loss_multimask = softmax_focal_loss(src_masks, target_masks)
        loss_multidice = dice_loss_softmax(
            src_masks, target_masks
        )
        if not self.pred_obj_scores:
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # loss_multiiou = iou_loss(
        #     src_masks,
        #     target_masks,
        #     ious,
        #     # num_objects,
        #     loss_on_multimask=True,
        #     use_l1_loss=self.iou_use_l1_loss,
        # )

        print(f'The multimask " {loss_multimask.sum()}')


        loss_mask = loss_multimask
        loss_dice = loss_multidice
        # loss_iou = loss_multiiou

        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True)
        losses["loss_class"] += loss_class.sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
    

def grad_vec(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    vec = [g.detach().flatten() for g in gs if g is not None]
    return torch.cat(vec) if vec else None



class MultiStepEntropyMinimization(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.step_count = 0

    def forward(self, outs_batch: List[Dict], target):
        print(f'The shapes of the inputs {target.shape}')

        losses = defaultdict(int)
        for frame_idx in range(len(outs_batch[0])):
            # For each frame
            frame_input = [view[frame_idx] for view in outs_batch]
            for i in frame_input:
                print(f'Length of inputs: {len(i), type(i) }')
            cur_losses = self._forward(frame_input, target[frame_idx])
            for k, v in cur_losses.items():
                print(f'The k: {k}, and v {v}')
                losses[k] += v


        return losses


    def _forward(self, outputs: List[Dict], target):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """


        con_masks = [output["multistep_pred_multimasks_high_res"] for output in outputs ]
        all_upscaled_embeddings = [output["multistep_upscaled_embeddings"] for output in outputs]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for i in con_masks:
            print(f'Length of : {len(i)}')
        for step_idx in range(1):
            # print('This is the second loop')
            con_masks_steps = [output[step_idx]for output in con_masks]
            print(f'Gets here too')
            self._update_losses(
                losses, con_masks_steps, all_upscaled_embeddings[step_idx], target
            )
            self.step_count += 1
        
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        print(f'here are the losses {losses}')
        return losses

    def _update_losses(
        self, losses, src_masks, upscaled_embeddings, target
    ):
        print(f'Prints here')
        # save_image(src_masks[0].float(), f'srcmasks{0}.png') 
        src_mask = src_masks[0].squeeze(0)
        upscaled_embedding = upscaled_embeddings[0]
        # print(f'The shape of inputs before entry to loss calcs: {src_masks.shape, target_masks.shape}')
        if len(src_mask.shape) == 3:
            src_mask = src_mask.unsqueeze(0)

        print(f'Gets here')
        loss_multimask, probability = softmax_entropy_loss(
            src_mask
        )
        # loss_iou = original_non_overlap_gap_loss(src_mask)
        # loss_multidice = consistency_loss(src_masks)
        con_masks = src_masks[1:]
        # print(len(f'Length of Con masks{len(con_masks)}'))
        # loss_multidice = cons_ce_multi(con_masks[1:], con_masks[0])
        # loss_class = compactness_loss(upscaled_embedding, src_mask)

        # loss_diversity_multi = cosine_diversity_loss(src_masks)
        # loss_multidice = softmax_focal_loss(con_masks[0], target)
        loss_mask = loss_multimask
        # loss_dice = loss_multidice
        # print(f'The loses are here: {loss_mask, loss_dice}')
        
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += probability
        print(f'Loss: {probability}, the thing, {loss_mask}')
        losses["loss_iou"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()
        losses["loss_class"] += torch.tensor(0.0, device=loss_mask.device, requires_grad=True).sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
    
import torch
import torch.nn.functional as F

# ----------------------------
# Core utilities
# ----------------------------
def to_one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """y: (B,H,W) or (B,D,H,W) -> (B,C,...) float one-hot"""
    return F.one_hot(y, num_classes=num_classes).permute(0, -1, *range(1, y.dim())).float()

def to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over channel dim=1 to get class probabilities."""
    return F.softmax(logits, dim=1)

def maybe_drop_background(p: torch.Tensor, y_oh: torch.Tensor, include_background: bool):
    """Optionally zero-out background (channel 0)."""
    if not include_background:
        p = p.clone(); y_oh = y_oh.clone()
        p[:, 0] = 0.0
        y_oh[:, 0] = 0.0
    return p, y_oh

# ----------------------------
# Laplacian kernels
# ----------------------------
def laplacian_kernel_2d(dtype, device) -> torch.Tensor:
    """3x3 4-neighborhood Laplacian."""
    K = torch.tensor([[0, 1, 0],
                      [1,-10, 1],
                      [0, 1, 0]], dtype=dtype, device=device)
    return K.view(1, 1, 3, 3)

def laplacian_kernel_3d(dtype, device) -> torch.Tensor:
    """3x3x3 6-neighborhood Laplacian (center -6, faces +1)."""
    K = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    K[1,1,1] = -6.0
    K[1,1,0] = K[1,1,2] = 1.0
    K[1,0,1] = K[1,2,1] = 1.0
    K[0,1,1] = K[2,1,1] = 1.0
    return K.view(1, 1, 3, 3, 3)

# ----------------------------
# Depthwise same-conv wrappers
# ----------------------------
def depthwise_laplacian_2d(x: torch.Tensor, pad_mode: str = "replicate") -> torch.Tensor:
    """x: (B,C,H,W) -> Laplacian response (B,C,H,W)"""
    B, C, H, W = x.shape
    K = laplacian_kernel_2d(x.dtype, x.device).repeat(C, 1, 1, 1)  # (C,1,3,3)
    pad = (1, 1, 1, 1)
    return F.conv2d(F.pad(x, pad, mode=pad_mode), K, groups=C)

def laplacian_boundary_loss_2d(
    logits: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = False,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Laplacian boundary loss for 2D: logits (B,C,H,W), y (B,H,W)."""
    assert logits.dim() == 4, "Expected logits (B,C,H,W) for 2D."
    B, C, H, W = logits.shape
    p = to_probs(logits).to(logits.dtype)         # (B,C,H,W)
    y_oh = to_one_hot(y, C).to(logits.dtype)      # (B,C,H,W)
    p, y_oh = maybe_drop_background(p, y_oh, include_background)
    p_lap  = depthwise_laplacian_2d(p, pad_mode)
    y_lap  = depthwise_laplacian_2d(y_oh, pad_mode)
    return huber(p_lap, y_lap)


def huber(x, y, beta=0.1, w=None):
    r = (x - y).abs()
    s = torch.where(r < beta, 0.5*(r**2)/beta, r - 0.5*beta)
    return (s if w is None else s*w).mean()

def consistency_boundary_loss(students, mask):
    losses = []
    for i, s in enumerate(students):
        losses.append(boundary_loss_from_labels(s.unsqueeze(0), mask, (1.0, 1.0 )))
    return torch.stack(losses).mean() 


import numpy as np
from scipy.ndimage import distance_transform_edt

def _sdm_binary_np(gt_bool: np.ndarray, spacing: tuple, band_mm: float | None) -> np.ndarray:
    # gt_bool: np.bool_, ndim=2 or 3. spacing=(dy,dx) or (dz,dy,dx) in mm
    fg = gt_bool.astype(bool)
    bg = ~fg
    dt_out = distance_transform_edt(bg, sampling=spacing)  # outside distance
    dt_in  = distance_transform_edt(fg, sampling=spacing)  # inside distance
    sdm = dt_out.astype(np.float32)
    sdm[fg] = -dt_in[fg]
    if band_mm is not None:
        sdm = np.clip(sdm, -band_mm, band_mm)
    return sdm

def boundary_loss_from_labels(
    logits_or_probs: torch.Tensor,     # [B,K,*]
    labels: torch.Tensor,              # [B,*] ints in {0..K-1}
    spacing_mm,                        # tuple for 2D/3D OR list/tuple of per-sample tuples
    include_background: bool = False,  # usually False
    band_mm: float | None = 15.0,      # clip SDM to ±band_mm to focus near boundary
    use_logits: bool = True,
    reduce: str = "mean",
    class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    x = logits_or_probs
    assert x.ndim in (4,5), "logits/probs must be [B,K,H,W] or [B,K,D,H,W]"
    
    probs = torch.softmax(x, dim=1).to(x.dtype)


    B, K = probs.shape[:2]
    assert labels.shape[0] == B and labels.shape[1:] == probs.shape[2:], "shape mismatch probs/labels"

    # classes to include
    cls_list = list(range(K)) if include_background else list(range(1, K))

    # normalize spacing per-sample
    if isinstance(spacing_mm, (tuple, list)) and isinstance(spacing_mm[0], (tuple, list)):
        spacings = spacing_mm
        assert len(spacings) == B, "provide spacing per sample"
    else:
        spacings = [tuple(spacing_mm)] * B

    # build SDMs per sample on CPU, then to torch
    sdm_tensors = []
    for b in range(B):
        lab_np = labels[b].detach().cpu().numpy()
        sdm_k = [_sdm_binary_np(lab_np == k, spacing=spacings[b], band_mm=band_mm) for k in cls_list]
        sdm_np = np.stack(sdm_k, axis=0)  # [K',*]
        sdm_tensors.append(torch.from_numpy(sdm_np))
    sdm = torch.stack(sdm_tensors, dim=0).to(probs.device, probs.dtype)  # [B,K',*]

    # select same classes in probs
    idx = torch.as_tensor(cls_list, device=probs.device, dtype=torch.long)
    probs_sel = probs.index_select(1, idx)

    loss_map = probs_sel * sdm
    if class_weights is not None:
        w = class_weights.to(loss_map.device, loss_map.dtype).view(1, -1, *([1] * (loss_map.ndim - 2)))
        loss_map = loss_map * w

    return loss_map.mean() if reduce == "mean" else loss_map.sum()

def count_classes_from_labels(labels: torch.Tensor, num_classes: int, ignore_index: int | None = None) -> torch.Tensor:
    """
    labels: [B,H,W] or [B,D,H,W], int64 in [0..num_classes-1] (or ignore_index)
    returns counts: [C] on the same device as labels
    """
    y = labels.view(-1).long()
    if ignore_index is not None:
        y = y[y != ignore_index]
    counts = torch.bincount(y, minlength=num_classes)
    return counts

def weights_effective_number(counts: torch.Tensor, beta: float = 0.999, clamp_max: float | None = None) -> torch.Tensor:
    """
    Cui et al., 'Class-Balanced Loss' (CVPR'19).
    counts: [C] (>=0). beta∈[0,1). Higher beta => stronger reweighting.
    Returns weights normalized to mean=1.
    """
    counts = counts.float().clamp_min(1)
    w = (1.0 - beta) / (1.0 - beta ** counts)
    w = w / w.mean()
    if clamp_max is not None:
        w = torch.clamp(w, max=clamp_max)
        w = w / w.mean()
    return w


def radial_order_constraint(probs, idx_ED=2, idx_NCR=1, idx_ET=3):
    B, C, *spatial = probs.shape
    coords = torch.stack(torch.meshgrid([torch.arange(s, device=probs.device) for s in spatial], indexing='ij'))
    center = (torch.tensor(spatial, device=probs.device) - 1) / 2
    dist = torch.linalg.norm(coords - center[:, None, None, None], dim=0)  # [H,W,D]
    
    # mean distance weighted by prob mass
    mean_d = [(probs[:, i] * dist).sum() / (probs[:, i].sum() + 1e-8) for i in [idx_ET, idx_NCR, idx_ED]]
    loss = torch.relu(mean_d[0] - mean_d[1]) + torch.relu(mean_d[1] - mean_d[2])
    return loss

# import torch
# import torch.nn.functional as F

# def radial_order_constraint(
#     probs: torch.Tensor,                 # [B,C,H,W] or [B,C,D,H,W], softmax probs
#     idx_ED = 2, idx_NCR = 1 , idx_ET = 2,
#     min_mass: float = 1e-4,              # skip a class if its total prob < min_mass
#     eps: float = 1e-8
# ) -> torch.Tensor:
#     """
#     Enforce radial ordering: mean radius(ET) <= mean radius(NCR) <= mean radius(ED),
#     measured w.r.t. the predicted tumor-core (TC=NCR+ET) center-of-mass.
#     Skips pairwise terms if a class is absent (mass < min_mass).
#     Returns a scalar (averaged over the number of active terms and batch).
#     """
#     B, C, *spatial = probs.shape
#     D = len(spatial)

#     # ---- coordinates grid [D, *spatial]
#     coords = torch.stack(
#         torch.meshgrid(
#             [torch.arange(s, device=probs.device, dtype=probs.dtype) for s in spatial],
#             indexing="ij"
#         ),
#         dim=0
#     )  # [D, *]

#     # ---- tumor-core center-of-mass (per batch)
#     p_tc = (probs[:, idx_NCR] + probs[:, idx_ET]).clamp_min(0)
#     mass_tc = p_tc.flatten(1).sum(dim=1).clamp_min(eps)  # [B]

#     # COM along each axis: sum(p_tc * coord) / sum(p_tc)
#     com = []
#     for d in range(D):
#         num = (p_tc * coords[d]).flatten(1).sum(dim=1)    # [B]
#         com_d = (num / mass_tc).view(B, *([1]*D))         # [B,1,1,(1)]
#         com.append(com_d)
#     # squared distance to COM
#     dist2 = 0.0
#     for d in range(D):
#         dist2 = dist2 + (coords[d] - com[d])**2
#     dist = torch.sqrt(dist2 + eps)                        # [*, spatial]
#     dist = dist.expand(B, *spatial)                       # [B,*]

#     # ---- per-class mean distances (prob-mass weighted)
#     def mean_radius(p_class):
#         m = p_class.flatten(1).sum(dim=1)                 # [B]
#         num = (p_class * dist).flatten(1).sum(dim=1)      # [B]
#         r = num / (m.clamp_min(eps))
#         present = (m >= min_mass).float()                 # [B]
#         return r, present

#     r_et, pres_et   = mean_radius(probs[:, idx_ET])
#     r_ncr, pres_ncr = mean_radius(probs[:, idx_NCR])
#     r_ed, pres_ed   = mean_radius(probs[:, idx_ED])

#     # ---- pairwise violations with presence gating
#     # ET <= NCR, NCR <= ED
#     v1 = torch.relu(r_et  - r_ncr) * (pres_et * pres_ncr)   # [B]
#     v2 = torch.relu(r_ncr - r_ed ) * (pres_ncr * pres_ed)   # [B]

#     # normalize by number of active terms per batch element
#     active = (pres_et * pres_ncr) + (pres_ncr * pres_ed)    # [B] in {0,1,2}
#     loss_per_b = torch.where(active > 0, (v1 + v2) / active.clamp_min(1.0), torch.zeros_like(v1))

#     return loss_per_b.mean()
