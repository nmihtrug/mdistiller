from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, AugTrainer, LoggerTrainer
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "aug": AugTrainer,
    "logger": LoggerTrainer,
}
