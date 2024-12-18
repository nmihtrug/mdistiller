import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def dkd_group_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    N, c = logits_student.shape
    
    if len(target.size()) > 1:
        label = torch.max(target, dim=1, keepdim=True)[1]
    else:
        label = target.view(len(target), 1)
    
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )

    binary_target = torch.tensor([[1.0, 0.0]] * pred_student.shape[0]).to(pred_student.device)
    
    bce = nn.BCELoss()
    bce_loss = bce(pred_student, binary_target)
    
    group1 = tckd_loss + bce_loss
    
    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_teacher_nag = logits_teacher[mask].reshape(N, -1)
    logits_student_nag = logits_student[mask].reshape(N, -1)
    
    pred_teacher_nag = F.softmax(
        logits_teacher_nag / temperature, dim=1
    )
    log_pred_student_nag = F.log_softmax(
        logits_student_nag / temperature, dim=1
    )
    
    nckd_loss = (
        F.kl_div(log_pred_student_nag, pred_teacher_nag, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    
    negative_labels_onehot = torch.zeros_like(log_pred_student_nag)
    
    ce_negative_loss = -(negative_labels_onehot * log_pred_student_nag).sum(dim=1).mean()
    
    group2 = nckd_loss + ce_negative_loss
    
    dkd_group_loss = alpha * group1 + beta * group2
    
    return alpha * group1, beta * group2, dkd_group_loss


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


class DKD_group(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD_group, self).__init__(student, teacher)
        self.alpha = cfg.DKD_group.ALPHA
        self.beta = cfg.DKD_group.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        self.loss_tckd, self.loss_nckd, loss_dkd_group = dkd_group_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        
        loss_dkd_group *= min(kwargs["epoch"] / self.warmup, 1.0) 
        
        losses_dict = {
            # "loss_ce": loss_ce,
            "loss_kd": loss_dkd_group,
        }
        return logits_student, losses_dict
    
    def get_losses_info(self):
        return {
            "loss_tckd": self.loss_tckd,
            "loss_nckd": self.loss_nckd,
        }