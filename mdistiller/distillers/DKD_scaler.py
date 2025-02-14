import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def sample_wise_scaler(logits_student, logits_teacher, target):
    pred_teacher = F.softmax(logits_teacher, dim=1)
    
    # ce loss between student and teacher
    ce_loss_teacher = F.cross_entropy(logits_student, pred_teacher, reduction='none')
    
    # ce loss between student and target
    ce_loss_target = F.cross_entropy(logits_student, target, reduction='none')
    
    # Compute focal weight
    focal_weight = torch.max(ce_loss_teacher / (ce_loss_target + 1e-7), torch.zeros_like(ce_loss_teacher))
    focal_weight = 1 - torch.exp(-focal_weight)
    
    return torch.mean(focal_weight)



def dkd_loss_with_scaler(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student, dim=1)
    pred_teacher = F.softmax(logits_teacher, dim=1)
    
    pred_loss_scaler = sample_wise_scaler(logits_student, logits_teacher, target)
    
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
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
    
    tckd_loss *= (1 - pred_loss_scaler)
    
    nckd_loss *= pred_loss_scaler
    
    dkd_loss = alpha * tckd_loss + beta * nckd_loss
    
    return alpha * tckd_loss, beta * nckd_loss, dkd_loss, pred_loss_scaler


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


class DKD_scaler(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD_scaler, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD_scaler.ALPHA
        self.beta = cfg.DKD_scaler.BETA
        self.temperature = cfg.DKD_scaler.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        self.loss_tckd, self.loss_nckd, loss_dkd, self.pred_loss_scaler = dkd_loss_with_scaler(
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
            "pred_loss_scaler": self.pred_loss_scaler,
        }
