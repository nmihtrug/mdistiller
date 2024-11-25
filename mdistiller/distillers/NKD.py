import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


# class NKDLoss(nn.Module):
#     """PyTorch version of NKD"""

#     def __init__(
#         self,
#         temp=1.0,
#         gamma=1.5,
#     ):
#         super(NKDLoss, self).__init__()

#         self.temp = temp
#         self.gamma = gamma
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, logit_s, logit_t, gt_label):

#         if len(gt_label.size()) > 1:
#             label = torch.max(gt_label, dim=1, keepdim=True)[1]
#         else:
#             label = gt_label.view(len(gt_label), 1)

#         # N*class
#         N, c = logit_s.shape
#         s_i = self.log_softmax(logit_s)
#         t_i = F.softmax(logit_t, dim=1)
#         # N*1
#         s_t = torch.gather(s_i, 1, label)
#         t_t = torch.gather(t_i, 1, label).detach()

#         loss_t = -(t_t * s_t).mean()

#         mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
#         logit_s = logit_s[mask].reshape(N, -1)
#         logit_t = logit_t[mask].reshape(N, -1)

#         # N*class
#         S_i = self.log_softmax(logit_s / self.temp)
#         T_i = F.softmax(logit_t / self.temp, dim=1)

#         loss_non = (T_i * S_i).sum(dim=1).mean()
#         loss_non = -self.gamma * (self.temp**2) * loss_non

#         return loss_t + loss_non

def NKD_loss(logit_s, logit_t, target, temperature, gamma):
    log_softmax = nn.LogSoftmax(dim=1)
    if len(target.size()) > 1:
        label = torch.max(target, dim=1, keepdim=True)[1]
    else:
        label = target.view(len(target), 1)

    # N*class
    N, c = logit_s.shape
    s_i = log_softmax(logit_s)
    t_i = F.softmax(logit_t, dim=1)
    # N*1
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()

    loss_t = -(t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
    logit_s = logit_s[mask].reshape(N, -1)
    logit_t = logit_t[mask].reshape(N, -1)

    # N*class
    S_i = log_softmax(logit_s / temperature)
    T_i = F.softmax(logit_t / temperature, dim=1)

    loss_non = (T_i * S_i).sum(dim=1).mean()
    loss_non = -gamma * (temperature**2) * loss_non

    return loss_t + loss_non

class NKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(NKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.NKD.CE_WEIGHT
        self.nkd_loss_weight = cfg.NKD.NKD_WEIGHT
        self.temperature = cfg.NKD.TEMPERATURE
        self.gamma = cfg.NKD.GAMMA

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
            
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_nkd = self.nkd_loss_weight * NKD_loss(logits_student, logits_teacher, target, self.temperature, self.gamma)
        losses_dict = {
            "loss_ce": loss_ce, 
            "loss_nkd": loss_nkd
        }
        return logits_student, losses_dict
