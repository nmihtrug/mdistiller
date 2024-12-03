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
}
