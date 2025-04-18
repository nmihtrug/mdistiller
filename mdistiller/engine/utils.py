import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))


    b_list = []
    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            
            output = distiller(image=image)
            if output.__class__ == list:
                output = output[0]
            
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    
    # print("B: ", np.mean(b_list), np.max(b_list), np.min(b_list), len(b_list))
    return top1.avg, top5.avg, losses.avg


def experiment(data_loader, model_s, model_t):
    batch_time, ce_loss_teacheres, losses, top1, top5 = [AverageMeter() for _ in range(5)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(data_loader)
    pbar = tqdm(range(num_iter))


    b_list = []
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(data_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            
            output_s = model_s(image=image)
            output_t = model_t(image=image)
            
            
            # target_logits = output[torch.arange(output.size(0)), target]

            # Mask to exclude target logits
            # mask = torch.ones_like(output, dtype=bool)
            # mask[torch.arange(output.size(0)), target] = False

            # # Get the maximum of non-target logits
            # non_target_max_logits = output.masked_fill(~mask, float('-inf')).max(dim=1).values

            # b_list.extend(target_logits.detach().cpu().numpy() - non_target_max_logits.detach().cpu().numpy())

            pred_t = F.softmax(output_t, dim=1)

            ce_loss_teacher = F.cross_entropy(output_s, pred_t, reduction='none')

            ce_loss_teacheres.update(ce_loss_teacher.cpu().detach().numpy().mean(), image.size(0))

            loss = criterion(output_s, target)
            acc1, acc5 = accuracy(output_s, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    
    # print("B: ", np.mean(b_list), np.max(b_list), np.min(b_list), len(b_list))
    print("CE Loss Teacher: ", ce_loss_teacheres.avg)
    print("CE Loss Target: ", losses.avg)
    return top1.avg, top5.avg, losses.avg

def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)
