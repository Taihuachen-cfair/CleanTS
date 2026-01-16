from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.common.torch_util import mask_fill, packed_attention_mask_VI
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import ResidualBlock


class CleanTSModule(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size of input
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        :param quantile_levels: Quantile levels for quantile-based operations (values in (0,1)).
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.upperRevin = (patch_size*max_seq_len-1)**0.5
        self.scaling = scaling
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)

        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj =ResidualBlock(
            input_dims=patch_size,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=self.num_quantiles * patch_size,
        )

    @staticmethod
    def compact_transform(
            x: torch.Tensor,
            threshold: float = 10.0
    ) -> torch.Tensor:
        """
        Compact transform: linear below threshold, logarithmic compression above threshold

        Args:
            x: Input value (tensor)
            threshold: Linear threshold, default 10.0

        Returns:
            Transformed value

        Formula:
            f(x) = { x                     if x ≤ threshold
                   { threshold + ln(1 + x - threshold) if x > threshold
        """

        result = torch.empty_like(x)

        # Linear region
        mask_linear = x <= threshold
        result[mask_linear] = x[mask_linear]

        # Logarithmic region
        mask_log = ~mask_linear
        result[mask_log] = threshold + torch.log1p(x[mask_log] - threshold)

        return result

    @staticmethod
    def inverse_compact_transform(
            y: torch.Tensor,
            threshold: float = 10.0
    ) -> torch.Tensor:
        """
        Inverse compact transform

        Args:
            y: Transformed value (tensor)
            threshold: Linear threshold, default 10.0

        Returns:
            Original value before transformation

        Formula:
            f⁻¹(y) = { y                     if y ≤ threshold
                     { threshold - 1 + exp(y - threshold) if y > threshold
        """
        result = torch.empty_like(y)

        # Linear region
        mask_linear = y <= threshold
        result[mask_linear] = y[mask_linear]

        # Exponential region
        mask_exp = ~mask_linear
        result[mask_exp] = threshold - 1 + torch.exp(y[mask_exp] - threshold)

        return result


    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        training_mode: bool = True,
    ):
        """
        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param training_mode: training status
        :return: quantile prediction
        """
        # RevIN
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale

        # compact_transform
        scaled_target = self.compact_transform(scaled_target, threshold=self.upperRevin)

        reprs = self.in_proj(scaled_target)
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

        attn_mask = packed_attention_mask_VI(sample_id,variate_id)

        reprs = self.encoder(
            masked_reprs,
            attn_mask,
            time_id=time_id,
            var_id=variate_id,
        )
        preds = self.out_proj(reprs)
        if training_mode:
            return preds, scaled_target
        else:
            # inverse_compact_transform before return
            return self.inverse_compact_transform(preds,threshold=self.upperRevin) * scale + loc