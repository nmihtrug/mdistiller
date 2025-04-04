import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss

from ._base import Distiller


def zscore(x):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6  # prevent divide-by-zero
    return (x - mean) / std

def logit_diff_loss(logits_student, logits_teacher, target, temperature):
    N, C = logits_student.shape

    # Normalize student and teacher logits
    logits_student = zscore(logits_student)
    teacher = zscore(logits_teacher)

    # Get target logits
    student_target_logits = logits_student[torch.arange(N), target].unsqueeze(1)
    teacher_target_logits = teacher[torch.arange(N), target].unsqueeze(1)

    # Compute logit margins
    student_margins = logits_student - student_target_logits  # (N, C)
    teacher_margins = logits_teacher - teacher_target_logits  # (N, C)

    # Remove target class from comparison (optional but common)
    mask = torch.ones_like(logits_student, dtype=bool)
    mask[torch.arange(N), target] = False
    student_margins = student_margins[mask].view(N, C - 1)
    teacher_margins = teacher_margins[mask].view(N, C - 1)

    # Compute MSE loss over margins
    loss = torch.mean((student_margins - teacher_margins) ** 2)
    return loss





class Logit_Diff(Distiller):
    def __init__(self, student, teacher, cfg):
        super(Logit_Diff, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.LD.CE_WEIGHT
        self.kd_loss_weight = cfg.LD.KD_WEIGHT
        self.temperature = cfg.LD.TEMPERATURE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_ld = self.kd_loss_weight * logit_diff_loss(
            logits_student,
            logits_teacher,
            target,
            self.temperature,
        )
         
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_ld,
        }
        return logits_student, losses_dict
    