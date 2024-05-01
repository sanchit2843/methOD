import torch
from lib.losses.dim_aware_loss import dim_aware_l1_loss
from lib.losses.centernet_loss import compute_heading_loss
from torch.nn import functional as F


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def match_proposals_with_targets(proposals, targets, iou_threshold=0.5):
    matched_indices = []
    ## iterate over batch
    for proposal, target in zip(proposals, targets):
        ## drop all zero rows
        proposal = proposal[proposal.sum(dim=1) != 0]
        target = target[target.sum(dim=1) != 0]
        if proposal.shape[0] == 0 or target.shape[0] == 0:
            matched_indices.append(torch.Tensor([-1] * proposal.shape[0]).long())
            continue
        iou = box_iou(proposal[:, 1:], target)
        # find the best match for ea1ch proposal
        best_match = iou.argmax(dim=1)
        ## keep only those that have iou > threshold, alot -1 for those that dont
        best_match[iou.max(dim=1).values < iou_threshold] = -1
        matched_indices.append(best_match)

    return matched_indices


def box_iou(boxes1, boxes2):
    """
    Calculate the IoU (Intersection over Union) between two sets of bounding boxes.

    Args:
        boxes1 (torch.Tensor): First set of bounding boxes (N, 4)
        boxes2 (torch.Tensor): Second set of bounding boxes (M, 4)

    Returns:
        iou (torch.Tensor): IoU matrix (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def compute_proposal_head_loss(preds, target):
    proposals = target["2d_bbox"]
    proposal_mask = target["mask_2d_pred"]
    ## remove rows with all zeros in b,n,5 tensor
    dim, rot_cls, rot_reg, loc = preds
    size_3d = target["size_3d"]
    heading_bin = target["heading_bin"]
    heading_res = target["heading_res"]
    localization_3d = target["3d_location"]
    mask_3d = target["mask_3d"].bool().unsqueeze(-1)
    gt_2d = target["gt_2d_bbox"]

    matched_indices = match_proposals_with_targets(proposals, gt_2d)

    matched_size_3d = []
    matched_heading_bin = []
    matched_heading_res = []
    matched_localization_3d = []
    matched_dim = []
    matched_rot_cls = []
    matched_rot_reg = []
    matched_loc = []

    dim_loss = 0
    rot_loss = 0
    localization_loss = 0

    for i, indices in enumerate(matched_indices):
        if torch.all(indices == -1):
            continue
        current_batch_size_3d = []
        current_batch_heading_bin = []
        current_batch_heading_res = []
        current_batch_localization_3d = []
        current_batch_dim = []
        current_batch_rot_cls = []
        current_batch_rot_reg = []
        current_batch_loc = []

        for idx, j in enumerate(indices):
            if j == -1:
                continue
            current_batch_size_3d.append(size_3d[i][j])
            current_batch_heading_bin.append(heading_bin[i][j])
            current_batch_heading_res.append(heading_res[i][j])
            current_batch_localization_3d.append(localization_3d[i][j])
            current_batch_dim.append(dim[i][idx])
            current_batch_rot_cls.append(rot_cls[i][idx])
            current_batch_rot_reg.append(rot_reg[i][idx])
            current_batch_loc.append(loc[i][idx])

        dim_loss += smooth_l1_loss(
            torch.stack(current_batch_dim), torch.stack(current_batch_size_3d)
        )

        rot_cls_loss = F.cross_entropy(
            torch.stack(current_batch_rot_cls),
            torch.stack(current_batch_heading_bin).view(-1),
            reduction="mean",
        )
        cls_one_hot = (
            torch.zeros(torch.stack(current_batch_heading_bin).shape[0], 12)
            .cuda()
            .scatter_(
                dim=1,
                index=torch.stack(current_batch_heading_bin).view(-1, 1),
                value=1,
            )
        )
        current_batch_rot_reg = torch.sum(
            torch.stack(current_batch_rot_reg) * cls_one_hot, 1
        )

        rot_reg_loss = F.l1_loss(
            current_batch_rot_reg, torch.stack(current_batch_heading_res).view(-1)
        )

        rot_loss += rot_cls_loss + rot_reg_loss
        localization_loss += smooth_l1_loss(
            torch.stack(current_batch_loc),
            torch.stack(current_batch_localization_3d),
        )
    if dim_loss + rot_loss + localization_loss == 0:
        return torch.tensor(0.0).cuda()
    total_loss = dim_loss + rot_loss + localization_loss
    return total_loss  # / len(matched_indices)

    # matched_size_3d = torch.cat(matched_size_3d)
    # matched_heading_bin = torch.cat(matched_heading_bin)
    # matched_heading_res = torch.cat(matched_heading_res)
    # matched_localization_3d = torch.cat(matched_localization_3d)
    # matched_dim = torch.cat(matched_dim)
    # matched_rot_cls = torch.cat(matched_rot_cls)
    # matched_rot_reg = torch.cat(matched_rot_reg)
    # matched_loc = torch.cat(matched_loc)

    # dim_loss = dim_aware_l1_loss(matched_dim, matched_size_3d)
    # rot_loss = compute_heading_loss(matched_rot_cls, matched_heading_bin)
    # localization_loss = smooth_l1_loss(
    #     matched_loc, matched_localization_3d, reduction="mean"
    # )
    # return dim_loss + rot_loss + localization_loss
