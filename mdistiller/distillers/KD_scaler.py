import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def sample_wise_scaler(pred_student, pred_teacher, target):
    # Compute cross-entropy losses for student and teacher
    ce_loss_teacher = F.cross_entropy(pred_student, pred_teacher, reduction='none')
    ce_loss_target = F.cross_entropy(pred_student, target, reduction='none')
    
    # Compute focal weight
    focal_weight = torch.max(ce_loss_teacher / (ce_loss_target + 1e-7), torch.zeros_like(ce_loss_teacher))
    focal_weight = 1 - torch.exp(-focal_weight)
    
    return torch.mean(focal_weight)


def kd_loss(logits_student, logits_teacher, target, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    
    pred_loss_scaler = sample_wise_scaler(F.softmax(logits_student, dim=1), F.softmax(logits_teacher, dim = 1), target)
    loss_kd *= pred_loss_scaler
    
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, target, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
