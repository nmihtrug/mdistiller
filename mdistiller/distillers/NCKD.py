import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def nckd_loss(logits_student, logits_teacher, target, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class NCKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(NCKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.NCKD.CE_WEIGHT
        self.beta = cfg.NCKD.BETA
        self.temperature = cfg.NCKD.T
        self.warmup = cfg.NCKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nckd = nckd_loss(
            logits_student,
            logits_teacher,
            target,
            self.beta,
            self.temperature,
        )
        
        loss_nckd *= min(kwargs["epoch"] / self.warmup, 1.0) 
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_nckd,
        }
        return logits_student, losses_dict