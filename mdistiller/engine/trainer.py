import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
from .dot import DistillationOrientedTrainer
from .optimizer import OptimGradAlign

import wandb

class BaseTrainer(object):
    def __init__(self, experiment_project, experiment_name, tags, distiller, train_loader, val_loader, num_classes, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1
        self.run_ok = False

        self.experiment_project = experiment_project
        self.experiment_name = experiment_name
        self.tags = tags

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
            if cfg.SOLVER.GA:
                optimizer = OptimGradAlign(optimizer)
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, epoch, log_dict):
        # tensorboard log
        for k, v in log_dict.items():
            self.tf_writer.add_scalar(k, v, epoch)
        self.tf_writer.flush()
        # wandb log
        
        if not self.run_ok:
            self.run_ok = True
            if self.cfg.LOG.WANDB:
                try:
                    wandb.init(project=self.experiment_project, name=self.experiment_name, tags=self.tags)
                except:
                    print(log_msg("Failed to use WANDB", "INFO"))
                    self.cfg.LOG.WANDB = False
        
        if self.cfg.LOG.WANDB:
            wandb.log(log_dict, step=epoch)
            
        if log_dict["test_acc_top1"] > self.best_acc:
            self.best_acc = log_dict["test_acc_top1"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.log
        with open(os.path.join(self.log_path, "worklog.log"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                # "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.5f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.log"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)

        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc_top1, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "current_lr": lr,
                "train_loss": train_meters["losses"].avg,
                "train_acc_top1": train_meters["top1"].avg,
                "train_acc_top5": train_meters["top5"].avg,
                "test_loss": test_loss,
                "test_acc_top1": test_acc_top1,
                "test_acc_top5": test_acc_top5,
            }
        )
        self.log(epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc_top1 >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        batch_size = image.size(0)
        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch, num_classes=self.num_classes)
        
        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        
        if self.cfg.SOLVER.GA:
            self.optimizer.ga_backward([loss])
        else:
            loss.backward()
            
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class LoggerTrainer(BaseTrainer):
    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)

        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "loss_tckd": AverageMeter(),
            "loss_nckd": AverageMeter(),
            # "pred_loss_scaler": AverageMeter(),
        }
        
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc_top1, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

        # log
        log_dict = OrderedDict(
            {
                "current_lr": lr,
                "train_loss": train_meters["losses"].avg,
                "train_acc_top1": train_meters["top1"].avg,
                "train_acc_top5": train_meters["top5"].avg,
                "test_loss": test_loss,
                "test_acc_top1": test_acc_top1,
                "test_acc_top5": test_acc_top5,
                # "pred_loss_scaler": train_meters["pred_loss_scaler"].avg,
                "loss_tckd": train_meters["loss_tckd"].avg,
                "loss_nckd": train_meters["loss_nckd"].avg,
            }
        )
        self.log(epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc_top1 >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )
    
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        batch_size = image.size(0)
        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch, num_classes=self.num_classes)
    
    
        losses_info_dict = self.distiller.module.get_losses_info()
        loss_tckd = losses_info_dict["loss_tckd"]
        loss_nckd = losses_info_dict["loss_nckd"]
        # pred_loss_scaler = losses_info_dict["pred_loss_scaler"]
        
        train_meters["loss_tckd"].update(loss_tckd.cpu().detach().numpy().mean(), batch_size)
        train_meters["loss_nckd"].update(loss_nckd.cpu().detach().numpy().mean(), batch_size)
        # train_meters["pred_loss_scaler"].update(pred_loss_scaler.cpu().detach().numpy().mean(), batch_size)
        

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg

class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class DOT(BaseTrainer):
    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.log"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDDOT(BaseTrainer):

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            m_task = cfg.SOLVER.MOMENTUM - cfg.SOLVER.DOT.DELTA
            m_kd = cfg.SOLVER.MOMENTUM + cfg.SOLVER.DOT.DELTA
            optimizer = DistillationOrientedTrainer(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=m_task,
                momentum_kd=m_kd,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.log"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # dot backward
        loss_ce, loss_kd = losses_dict['loss_ce'].mean(), losses_dict['loss_kd'].mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        self.optimizer.step_kd()
        self.optimizer.zero_grad(set_to_none=True)
        loss_ce.backward()
        # self.optimizer.step((1 - epoch / 240.))
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update((loss_ce + loss_kd).cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class AugTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image_weak, image_strong = image
        image_weak, image_strong = image_weak.float(), image_strong.float()
        image_weak, image_strong = image_weak.cuda(non_blocking=True), image_strong.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image_weak=image_weak, image_strong=image_strong, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image_weak.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg