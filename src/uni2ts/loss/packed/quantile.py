import torch

from ._base import PackedQuantileLoss

class PackedQuantileMAELoss(PackedQuantileLoss):
    error_func = torch.nn.L1Loss(reduction="none")