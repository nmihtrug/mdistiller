import torch
import torch.nn as nn
from torch.nn import KLDivLoss
import torch.nn.functional as F
import math

from ._base import Distiller

class BinaryKL_loss(nn.Module):
    def __init__(self, temperature=2, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        self.criterion = KLDivLoss(reduction="none")

    def forward(self, student, teacher):
        N, _ = student.shape
        student = torch.sigmoid(student / self.temperature)
        teacher = torch.sigmoid(teacher / self.temperature)
        student = torch.clamp(student, min=self.eps, max=1-self.eps)
        teacher = torch.clamp(teacher, min=self.eps, max=1-self.eps)
        loss = self.criterion(torch.log(student), teacher) + self.criterion(torch.log(1 - student), 1 - teacher)
        return self.temperature ** 2 * loss.sum() / N


class BinaryKL(Distiller):

    def __init__(self, student, teacher, cfg):
        super(BinaryKL, self).__init__(student, teacher)
        self.temperature = cfg.BinaryKL.TEMPERATURE
        self.ce_loss_weight = cfg.BinaryKL.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.BinaryKL.LOSS.BinaryKL_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        binaryKL_loss = BinaryKL_loss(self.temperature)
        loss_kd = self.kd_loss_weight * binaryKL_loss(logits_student, logits_teacher)
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict