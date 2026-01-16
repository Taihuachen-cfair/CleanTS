#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import abc
from typing import Any, Optional, Callable

import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.common.core import abstract_class_property
from uni2ts.common.torch_util import safe_div


class PackedLoss(abc.ABC):
    """
    Abstract base class for loss functions supporting packed inputs.
    Subclasses should implement the _loss_func method which computes the loss function per token.
    """

    def __call__(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]] = None,
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, ""]:
        """
        :param pred: predictions
        :param target: target labels
        :param prediction_mask: 1 for predictions, 0 for non-predictions
        :param observed_mask: 1 for observed values, 0 for non-observed values
        :param sample_id: integer array representing the sample id
        :param variate_id: integer array representing the variate id
        :return: loss
        """
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        loss = self._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return self.reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...

    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
    ) -> Float[torch.Tensor, ""]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        # 预先mask以剪枝梯度图
        loss = loss * mask
        # 同组 token 的预测点总数
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        # 同组 token 的预测 token 总数（仅限预测 token）
        nobs = reduce(
            id_mask * rearrange(prediction_mask, "... seq -> ... 1 seq"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        ) * prediction_mask.unsqueeze(-1)
        nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()
        loss = safe_div(loss, tobs * nobs)
        return loss.sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class PackedLossOnToken(abc.ABC):
    """
    Abstract base class for loss functions supporting packed inputs.
    Subclasses should implement the _loss_func method which computes the loss function per token.
    """

    def __call__(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, ""]:
        """
        :param pred: predictions
        :param target: target labels
        :param prediction_mask: 1 for predictions, 0 for non-predictions
        :param sample_id: integer array representing the sample id
        :param variate_id: integer array representing the variate id
        :return: loss
        """
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        loss = self._loss_func(
            pred, target, prediction_mask, sample_id, variate_id
        )
        return self.reduce_loss(
            loss, prediction_mask, sample_id, variate_id
        )

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len"]: ...

    @staticmethod
    def reduce_loss(
            loss: Float[torch.Tensor, "*batch seq_len"],
            prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
            sample_id: Int[torch.Tensor, "*batch seq_len"],
            variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, ""]:
        """
        loss:        [..., seq_len]
        prediction_mask: same shape as loss
        sample_id:   same shape as loss
        variate_id:  same shape as loss
        """
        # 只保留有效位置
        loss = loss * prediction_mask
        valid_mask = prediction_mask.bool()

        # 展平
        flat_loss = loss[valid_mask]
        flat_sample_id = sample_id[valid_mask]
        flat_variate_id = variate_id[valid_mask]

        # 最小组：同 sample_id 且同 variate_id
        min_groups = torch.stack([flat_sample_id, flat_variate_id], dim=1)  # [N, 2]

        # 找到唯一最小组及其索引
        unique_min_groups, min_group_idx = torch.unique(min_groups, return_inverse=True, dim=0)

        # 每个最小组的平均 loss
        min_group_sum = (torch.zeros(unique_min_groups.shape[0], device=loss.device)
                         .scatter_add_(0, min_group_idx, flat_loss))
        min_group_count = (torch.zeros(unique_min_groups.shape[0], device=loss.device)
                           .scatter_add_(0, min_group_idx,torch.ones_like(flat_loss)))
        min_group_mean = safe_div(min_group_sum, min_group_count)

        # 大组：同 sample_id
        unique_sample_ids, sample_group_idx = torch.unique(unique_min_groups[:, 0], return_inverse=True)

        # 每个大组的最小组均值的平均
        sample_group_sum = (torch.zeros(unique_sample_ids.shape[0], device=loss.device)
                            .scatter_add_(0, sample_group_idx,min_group_mean))
        sample_group_count = (torch.zeros(unique_sample_ids.shape[0], device=loss.device)
                              .scatter_add_(0,sample_group_idx,torch.ones_like(min_group_mean)))
        sample_group_mean = safe_div(sample_group_sum, sample_group_count)

        # 所有大组均值的平均
        final_loss = sample_group_mean.mean()

        return final_loss

class PackedPointLoss(PackedLoss):
    """Abstract base class for loss functions on point forecasts."""

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...


class PackedDistributionLoss(PackedLoss):
    """Abstract base class for loss functions on probabilistic (distribution) forecasts."""

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Distribution,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...

class PackedClassifyLoss(PackedLossOnToken):
    """Abstract base class for loss functions on probabilistic (distribution) forecasts."""

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len" ],
        target: Float[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len"]: ...

class PackedQuantileLoss(PackedLoss):
    """
    Base class for quantile forecast loss functions.

    This class defines the quantile regression loss commonly used in
    probabilistic forecasting tasks. The child classes only need to specify
    an `error_func`, such as L1 (MAE) or L2 (MSE).
    """

    # ---- Abstract attribute: defines how errors are computed ----
    @abstract_class_property("error_func")
    def error_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Element-wise error function, e.g., L1Loss or MSELoss."""
        raise NotImplementedError

    def __init__(self, quantile_levels=None):
        super().__init__()
        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantile_levels = quantile_levels

    # noinspection PyUnusedLocal
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len num_quantiles*patch_size"],
        target: Float[torch.Tensor, "*batch seq_len patch_size"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch_size"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len patch_size"]:

        quantile_levels = torch.tensor(self.quantile_levels, device=pred.device).view(
            1, 1, -1, 1
        )

        pred = rearrange(
            pred,
            "... (num_quantiles patch_size) -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )

        target = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.quantile_levels),
        )

        errors = self.error_func(pred, target)
        indicator = target > pred

        quantile_loss = torch.where(
            indicator, quantile_levels * errors, (1 - quantile_levels) * errors
        )

        # Aggregate over quantiles
        return quantile_loss.mean(dim=-2)