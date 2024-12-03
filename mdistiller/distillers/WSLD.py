import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def wsld_loss(logits_student, logits_teacher, target, temperature, num_classes):
    s_input_for_softmax = logits_student / temperature
    t_input_for_softmax = logits_teacher / temperature

    t_soft_label = F.softmax(t_input_for_softmax, dim=1)
    
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax_loss = - torch.sum(t_soft_label * logsoftmax(s_input_for_softmax), 1, keepdim=True)

    logits_student_auto = logits_student.detach()
    logits_teacher_auto = logits_teacher.detach()
    log_softmax_s = logsoftmax(logits_student_auto)
    log_softmax_t = logsoftmax(logits_teacher_auto)
    
    one_hot_label = F.one_hot(target, num_classes=num_classes).float()
    softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
    softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

    focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
    ratio_lower = torch.zeros(1).cuda()
    focal_weight = torch.max(focal_weight, ratio_lower)
    focal_weight = 1 - torch.exp(- focal_weight)
    softmax_loss = focal_weight * softmax_loss

    loss_wsld = (temperature ** 2) * torch.mean(softmax_loss)
    return loss_wsld


class WSLD(Distiller):
    """Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective"""

    def __init__(self, student, teacher, cfg):
        super(WSLD, self).__init__(student, teacher)
        self.temperature = cfg.WSLD.TEMPERATURE
        self.ce_loss_weight = cfg.WSLD.CE_WEIGHT
        self.kd_loss_weight = cfg.WSLD.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_wsld = self.kd_loss_weight * wsld_loss(
            logits_student, logits_teacher, target, self.temperature, kwargs['num_classes']
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_wsld,
        }
        return logits_student, losses_dict
