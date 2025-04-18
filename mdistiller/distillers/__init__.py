from ._base import Vanilla
from .KD import KD
from .AT import AT
from .OFD import OFD
from .RKD import RKD
from .FitNet import FitNet
from .KDSVD import KDSVD
from .CRD import CRD
from .NST import NST
from .PKT import PKT
from .SP import SP
from .VID import VID
from .ReviewKD import ReviewKD
from .DKD import DKD
from .MLKD import MLKD
from .NKD import NKD
from .KD_CE import KD_CE
from .GDKD import GDKD
from .WSLD import WSLD
from .DKD_WSLD import DKD_WSLD
from .DKD_new import DKD_new
from .DKD_scaler import DKD_scaler
from .KD_scaler import KD_scaler
from .NCKD import NCKD
from .NCKD_BCE import NCKD_BCE
from .DKD_group import DKD_group
from .DHKD import DHKD
from .BinaryKL import BinaryKL
from .BinaryKL_norm import BinaryKL_Norm
from .BinaryKLNorm import BinaryKLNorm
from .logit_diff import Logit_Diff

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "AT": AT,
    "OFD": OFD,
    "RKD": RKD,
    "FITNET": FitNet,
    "KDSVD": KDSVD,
    "CRD": CRD,
    "NST": NST,
    "PKT": PKT,
    "SP": SP,
    "VID": VID,
    "REVIEWKD": ReviewKD,
    "DKD": DKD,
    "MLKD": MLKD,
    "NKD": NKD,
    "KD_CE": KD_CE,
    "GDKD": GDKD,
    "WSLD": WSLD,
    "DKD_WSLD": DKD_WSLD,
    "DKD_new": DKD_new,
    "DKD_scaler": DKD_scaler,
    "KD_scaler": KD_scaler,
    "NCKD": NCKD,
    "NCKD_BCE": NCKD_BCE,
    "DKD_group": DKD_group,
    "DHKD": DHKD,
    "BinaryKL": BinaryKL,
    "BinaryKLNorm": BinaryKLNorm,
    "BinaryKL_Norm": BinaryKL_Norm,
    "Logit_Diff": Logit_Diff,
}
