import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .BinaryKLNorm import BinaryKLNorm

def dhkd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class DHKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(DHKD, self).__init__(student, teacher)
        self.temperature = cfg.DHKD.TEMPERATURE
        self.ce_loss_weight = cfg.DHKD.CE_WEIGHT
        self.dhkd_loss_weight = cfg.DHKD.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student[0], target)
        
        dhkd_loss = BinaryKLNorm(temperature=self.temperature)
        
        loss_dhkd = self.dhkd_loss_weight * dhkd_loss(
            logits_student[1], logits_teacher
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_dhkd": loss_dhkd,
        }
        return logits_student[0], losses_dict
