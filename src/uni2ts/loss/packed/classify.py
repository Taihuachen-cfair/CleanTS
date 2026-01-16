import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from ._base import PackedClassifyLoss


class PackedCrossLoss(PackedClassifyLoss):
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len num_classes"],  # [B, L, num_classes]
        target: Int[torch.Tensor, "*batch seq_len"],              # [B, L]
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len"]:
        B, L, C = pred.shape
        loss = F.cross_entropy(pred.view(-1, C), target.view(-1), reduction='none')
        loss = loss.view(B, L)
        return loss
