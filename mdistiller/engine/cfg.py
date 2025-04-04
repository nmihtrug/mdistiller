from yacs.config import CfgNode as CN
from .utils import log_msg


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Experiment
CFG.EXPERIMENT = CN()
CFG.EXPERIMENT.PROJECT = "distill"
CFG.EXPERIMENT.NAME = ""
CFG.EXPERIMENT.TAG = "default"

# Dataset
CFG.DATASET = CN()
CFG.DATASET.TYPE = "cifar100"
CFG.DATASET.NUM_WORKERS = 2
CFG.DATASET.TEST = CN()
CFG.DATASET.TEST.BATCH_SIZE = 64

# Distiller
CFG.DISTILLER = CN()
CFG.DISTILLER.TYPE = "NONE"  # Vanilla as default
CFG.DISTILLER.TEACHER = "ResNet50"
CFG.DISTILLER.STUDENT = "resnet32"
CFG.DISTILLER.AUX_CLASSIFIER = False
CFG.DISTILLER.AUX_CLASSIFIER_LINEAR = True

# Solver
CFG.SOLVER = CN()
CFG.SOLVER.TRAINER = "base"
CFG.SOLVER.BATCH_SIZE = 64
CFG.SOLVER.EPOCHS = 240
CFG.SOLVER.LR = 0.05
CFG.SOLVER.LR_DECAY_STAGES = [150, 180, 210]
CFG.SOLVER.LR_DECAY_RATE = 0.1
CFG.SOLVER.WEIGHT_DECAY = 0.0001
CFG.SOLVER.MOMENTUM = 0.9
CFG.SOLVER.TYPE = "SGD"
CFG.SOLVER.GA = False

# Log
CFG.LOG = CN()
CFG.LOG.TENSORBOARD_FREQ = 500
CFG.LOG.SAVE_CHECKPOINT_FREQ = 40
CFG.LOG.PREFIX = "./output"
CFG.LOG.WANDB = True

# Distillation Methods

# KD CFG
CFG.KD = CN()
CFG.KD.TEMPERATURE = 4
CFG.KD.LOSS = CN()
CFG.KD.LOSS.CE_WEIGHT = 0.1
CFG.KD.LOSS.KD_WEIGHT = 0.9

# AT CFG
CFG.AT = CN()
CFG.AT.P = 2
CFG.AT.LOSS = CN()
CFG.AT.LOSS.CE_WEIGHT = 1.0
CFG.AT.LOSS.FEAT_WEIGHT = 1000.0

# RKD CFG
CFG.RKD = CN()
CFG.RKD.DISTANCE_WEIGHT = 25
CFG.RKD.ANGLE_WEIGHT = 50
CFG.RKD.LOSS = CN()
CFG.RKD.LOSS.CE_WEIGHT = 1.0
CFG.RKD.LOSS.FEAT_WEIGHT = 1.0
CFG.RKD.PDIST = CN()
CFG.RKD.PDIST.EPSILON = 1e-12
CFG.RKD.PDIST.SQUARED = False

# FITNET CFG
CFG.FITNET = CN()
CFG.FITNET.HINT_LAYER = 2  # (0, 1, 2, 3, 4)
CFG.FITNET.INPUT_SIZE = (32, 32)
CFG.FITNET.LOSS = CN()
CFG.FITNET.LOSS.CE_WEIGHT = 1.0
CFG.FITNET.LOSS.FEAT_WEIGHT = 100.0

# KDSVD CFG
CFG.KDSVD = CN()
CFG.KDSVD.K = 1
CFG.KDSVD.LOSS = CN()
CFG.KDSVD.LOSS.CE_WEIGHT = 1.0
CFG.KDSVD.LOSS.FEAT_WEIGHT = 1.0

# OFD CFG
CFG.OFD = CN()
CFG.OFD.LOSS = CN()
CFG.OFD.LOSS.CE_WEIGHT = 1.0
CFG.OFD.LOSS.FEAT_WEIGHT = 0.001
CFG.OFD.CONNECTOR = CN()
CFG.OFD.CONNECTOR.KERNEL_SIZE = 1

# NST CFG
CFG.NST = CN()
CFG.NST.LOSS = CN()
CFG.NST.LOSS.CE_WEIGHT = 1.0
CFG.NST.LOSS.FEAT_WEIGHT = 50.0

# PKT CFG
CFG.PKT = CN()
CFG.PKT.LOSS = CN()
CFG.PKT.LOSS.CE_WEIGHT = 1.0
CFG.PKT.LOSS.FEAT_WEIGHT = 30000.0

# SP CFG
CFG.SP = CN()
CFG.SP.LOSS = CN()
CFG.SP.LOSS.CE_WEIGHT = 1.0
CFG.SP.LOSS.FEAT_WEIGHT = 3000.0

# VID CFG
CFG.VID = CN()
CFG.VID.LOSS = CN()
CFG.VID.LOSS.CE_WEIGHT = 1.0
CFG.VID.LOSS.FEAT_WEIGHT = 1.0
CFG.VID.EPS = 1e-5
CFG.VID.INIT_PRED_VAR = 5.0
CFG.VID.INPUT_SIZE = (32, 32)

# CRD CFG
CFG.CRD = CN()
CFG.CRD.MODE = "exact"  # ("exact", "relax")
CFG.CRD.FEAT = CN()
CFG.CRD.FEAT.DIM = 128
CFG.CRD.FEAT.STUDENT_DIM = 256
CFG.CRD.FEAT.TEACHER_DIM = 256
CFG.CRD.LOSS = CN()
CFG.CRD.LOSS.CE_WEIGHT = 1.0
CFG.CRD.LOSS.FEAT_WEIGHT = 0.8
CFG.CRD.NCE = CN()
CFG.CRD.NCE.K = 16384
CFG.CRD.NCE.MOMENTUM = 0.5
CFG.CRD.NCE.TEMPERATURE = 0.07

# ReviewKD CFG
CFG.REVIEWKD = CN()
CFG.REVIEWKD.CE_WEIGHT = 1.0
CFG.REVIEWKD.REVIEWKD_WEIGHT = 1.0
CFG.REVIEWKD.WARMUP_EPOCHS = 20
CFG.REVIEWKD.SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.OUT_SHAPES = [1, 8, 16, 32]
CFG.REVIEWKD.IN_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.OUT_CHANNELS = [64, 128, 256, 256]
CFG.REVIEWKD.MAX_MID_CHANNEL = 512
CFG.REVIEWKD.STU_PREACT = False

# DKD(Decoupled Knowledge Distillation) CFG
CFG.DKD = CN()
CFG.DKD.CE_WEIGHT = 1.0
CFG.DKD.ALPHA = 1.0
CFG.DKD.BETA = 8.0
CFG.DKD.T = 4.0
CFG.DKD.WARMUP = 20


# DOT CFG
CFG.SOLVER.DOT = CN()
CFG.SOLVER.DOT.DELTA = 0.075


# NKD CFG
CFG.NKD = CN()
CFG.NKD.CE_WEIGHT = 1.0
CFG.NKD.NKD_WEIGHT = 1.0
CFG.NKD.TEMPERATURE = 4
CFG.NKD.GAMMA = 1.5


# GDKD CFG
CFG.GDKD = CN()
CFG.GDKD.CE_WEIGHT = 1.0
CFG.GDKD.W0 = 1.0
CFG.GDKD.W1 = 1.0
CFG.GDKD.W2 = 8.0
CFG.GDKD.TOPK = 5
CFG.GDKD.STRATEGY = "best" # or "worst"
CFG.GDKD.KL_TYPE = "forward"
CFG.GDKD.T = 4.0
CFG.GDKD.WARMUP = 20

# WSLD CFG
CFG.WSLD = CN()
CFG.WSLD.CE_WEIGHT = 1.0
CFG.WSLD.KD_WEIGHT = 2.25
CFG.WSLD.TEMPERATURE = 4

# DKD_scaler CFG
CFG.DKD_scaler = CN()
CFG.DKD_scaler.ALPHA = 1.0
CFG.DKD_scaler.BETA = 8.0
CFG.DKD_scaler.T = 4.0

# NCKD CFG
CFG.NCKD = CN()
CFG.NCKD.CE_WEIGHT = 1.0
CFG.NCKD.BETA = 1.0
CFG.NCKD.T = 4
CFG.NCKD.WARMUP = 20

# NCKD_BCE CFG
CFG.NCKD_BCE = CN()
CFG.NCKD_BCE.CE_WEIGHT = 0.0
CFG.NCKD_BCE.ALPHA = 1.0
CFG.NCKD_BCE.BETA = 1.0
CFG.NCKD_BCE.T = 4
CFG.NCKD_BCE.WARMUP = 20

# DKD_group CFG
CFG.DKD_group = CN()
CFG.DKD_group.ALPHA = 1.0
CFG.DKD_group.BETA = 8.0

# DHKD CFG
CFG.DHKD = CN()
CFG.DHKD.CE_WEIGHT = 1.0
CFG.DHKD.KD_WEIGHT = 1.0
CFG.DHKD.TEMPERATURE = 2

# BinaryKL CFG
CFG.BinaryKL = CN()
CFG.BinaryKL.TEMPERATURE = 2
CFG.BinaryKL.LOSS = CN()
CFG.BinaryKL.LOSS.CE_WEIGHT = 0.0
CFG.BinaryKL.LOSS.BinaryKL_WEIGHT = 1.0

# Logisd Diff CFG
CFG.LD = CN()
CFG.LD.CE_WEIGHT = 1.0
CFG.LD.KD_WEIGHT = 1.0
CFG.LD.TEMPERATURE = 2