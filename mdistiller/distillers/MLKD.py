import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller

def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MLKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MLKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        batch_size, class_num = logits_student.shape

        pred_teacher = F.softmax(logits_teacher.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * ((kd_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student,
            logits_teacher,
            3.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student,
            logits_teacher,
            5.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student,
            logits_teacher,
            2.0,
            # reduce=False
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student,
            logits_teacher,
            6.0,
            # reduce=False
        ) * mask).mean())

        loss_cc = self.kd_loss_weight * ((cc_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student,
            logits_teacher,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student,
            logits_teacher,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student,
            logits_teacher,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student,
            logits_teacher,
            6.0,
        ) * class_conf_mask).mean())

        loss_bc = self.kd_loss_weight * ((bc_loss(
            logits_student,
            logits_teacher,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student,
            logits_teacher,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student,
            logits_teacher,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student,
            logits_teacher,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student,
            logits_teacher,
            6.0,
        ) * mask).mean())
        
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_cc": loss_cc,
            "loss_bc": loss_bc
        }
        return logits_student, losses_dict
