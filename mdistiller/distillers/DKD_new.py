import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def wsld_scaler(logits_student, logits_teacher, target):
    # Compute cross-entropy losses for student and teacher
    ce_loss_student = F.cross_entropy(logits_student, target, reduction='none')
    ce_loss_teacher = F.cross_entropy(logits_teacher, target, reduction='none')
    
    # Compute focal weight
    focal_weight = torch.max(ce_loss_student / (ce_loss_teacher + 1e-7), torch.zeros_like(ce_loss_student))
    focal_weight = 1 - torch.exp(-focal_weight)
    
    return torch.mean(focal_weight)

def dkd_loss_with_wsld(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    
    wsld_weight = wsld_scaler(logits_student, logits_teacher, target)
    
    
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    
    tckd_loss *= (1 - wsld_weight)
    
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
    
    nckd_loss *= wsld_weight
    
    dkd_loss = alpha * tckd_loss + beta * nckd_loss
    
    return alpha * tckd_loss, beta * nckd_loss, dkd_loss


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


class DKD_new(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD_new, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        self.loss_tckd, self.loss_nckd, loss_dkd = dkd_loss_with_wsld(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        
        loss_dkd *= min(kwargs["epoch"] / self.warmup, 1.0) 
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
    
    def get_losses_info(self):
        return {
            "loss_tckd": self.loss_tckd,
            "loss_nckd": self.loss_nckd,
        }