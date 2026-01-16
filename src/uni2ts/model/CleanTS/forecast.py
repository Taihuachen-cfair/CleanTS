import math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Generator, Optional, List

import lightning as L
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from gluonts.itertools import batcher
from gluonts.model import QuantileForecast
from jaxtyping import Bool, Float, Int

from .module import CleanTSModule
from ...transform import DummyValueImputation


class CleanTSForecast(L.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        prediction_length: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[CleanTSModule] = None,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        if module_kwargs and "attn_dropout_p" in module_kwargs:
            module_kwargs["attn_dropout_p"] = 0
        if module_kwargs and "dropout_p" in module_kwargs:
            module_kwargs["dropout_p"] = 0

        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = CleanTSModule(**module_kwargs) if module is None else module

    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Generator["CleanTSForecast", None, None]:
        kwargs = {
            "prediction_length": prediction_length,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
        }
        old_hparams = deepcopy(self.hparams)
        for kw, arg in kwargs.items():
            if arg is not None:
                self.hparams[kw] = arg

        yield self

        for kw in kwargs:
            self.hparams[kw] = old_hparams[kw]

    def create_predictor(self, batch_size: int):
        class CleanTSQuantilePredictor:
            def __init__(
                    self,
                    model: CleanTSForecast,
                    batch_size: int = 2048,
                    quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    device: str = 'auto',
            ):
                self.batch_size = batch_size
                self.quantile_levels = quantile_levels
                self.device = self.get_device(device)
                self.model = model.to(self.device)

            def predict(self, test_data_input):
                # Generate forecast samples
                forecast_quantiles = []
                for batch in (batcher(test_data_input, batch_size=self.batch_size)):
                    past_target = [entry["target"] for entry in batch]
                    forecasts = self.model.predict(
                        past_target)  # full_forecasts shape: (batch num_quantiles future_time #tgt)
                    forecast_quantiles.append(forecasts)
                forecast_quantiles = np.concatenate(forecast_quantiles)

                # Convert forecast samples into gluonts QuantileForecast objects
                quantile_forecasts = []
                for item, ts in zip(forecast_quantiles, test_data_input):
                    forecast_start_date = ts["start"] + len(ts["target"])
                    quantile_forecasts.append(
                        QuantileForecast(
                            item_id=ts["item_id"],
                            forecast_arrays=item,
                            start_date=forecast_start_date,
                            forecast_keys=list(map(str, self.quantile_levels)))
                    )
                return quantile_forecasts

            @staticmethod
            def get_device(device="auto"):
                if device == "auto":
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")
                else:
                    device = torch.device(device)
                return device
        return CleanTSQuantilePredictor(model=self, batch_size=batch_size)

    def predict(
        self,
        past_target: List[Float[np.ndarray, "past_time tgt"]],
        feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
        past_feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
    ) -> Float[np.ndarray, "batch num_quantiles future_time *tgt"]:

        # only support univariate forecast now
        # implementation refer to https://github.com/awslabs/gluonts/blob/v0.15.x/src/gluonts/transform/split.py#L523
        data_entry = {"past_target": past_target, "feat_dynamic_real": feat_dynamic_real,
                      "past_feat_dynamic_real": past_feat_dynamic_real,
                      "past_observed_target": [~np.isnan(x) for x in past_target]}

        if feat_dynamic_real:
            data_entry["observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in feat_dynamic_real
            ]
        else:
            data_entry["observed_feat_dynamic_real"] = None

        if past_feat_dynamic_real:
            data_entry["past_observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in past_feat_dynamic_real
            ]
        else:
            data_entry["past_observed_feat_dynamic_real"] = None

        # check and use the same imputation strategy as pretraining
        impute = DummyValueImputation()

        def process_sample(sample):
            arr = np.asarray(sample)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
                arr = impute(arr)
            return arr

        for key, value in data_entry.items():
            if value is not None:
                data_entry[key] = [process_sample(sample) for sample in value]

        data_entry["past_is_pad"] = np.zeros(
            (len(data_entry["past_target"]), self.hparams.context_length), dtype=bool
        )

        # pad or slice context data
        for key in data_entry:
            if data_entry[key] is not None and isinstance(data_entry[key], list):
                for idx in range(len(data_entry[key])):
                    if data_entry[key][idx].shape[0] > self.hparams.context_length:
                        data_entry[key][idx] = data_entry[key][idx][
                            -self.hparams.context_length :, :
                        ]
                    else:
                        # print(key, idx, data_entry[key][idx].shape, data_entry[key][idx][0].shape)
                        pad_length = (
                            self.hparams.context_length - data_entry[key][idx].shape[0]
                        )
                        pad_block = np.full(
                            (pad_length, 1),
                            data_entry[key][idx][0],
                            dtype=data_entry[key][idx].dtype,
                        )  # alternative: padding with 0
                        data_entry[key][idx] = np.concatenate(
                            [pad_block, data_entry[key][idx]], axis=0
                        )
                        if key == "past_target":
                            data_entry["past_is_pad"][idx, :pad_length] = True

        for k in ["past_target", "feat_dynamic_real", "past_feat_dynamic_real"]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.float32
                )

        for k in [
            "past_observed_target",
            "observed_feat_dynamic_real",
            "past_observed_feat_dynamic_real",
            "past_is_pad",
        ]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=self.device, dtype=torch.bool
                )

        with torch.no_grad():
            predictions = self(**data_entry).detach().cpu().numpy()
        return predictions

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.hparams.context_length / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.hparams.prediction_length / patch_size)

    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.module.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        # get predictions
        # preds: Float[torch.Tensor, "*batch seq_len self.num_quantiles * patch_size"]
        preds = self.module(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=False,
        )
        preds = rearrange(
            preds,
            'batch seq_len (num_quantiles patch) -> batch seq_len num_quantiles patch',
            num_quantiles=9,
        )
        return self._format_preds(
            num_quantiles=9,
            patch_size=self.module.patch_size,
            preds=preds,
        )

    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: Float[torch.Tensor, "batch combine_seq num_quantiles patch"],
    ) -> Float[torch.Tensor, "batch num_quantiles future_time *tgt"]:
        start = self.context_token_length(patch_size)
        end = start + self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=1,
        )[..., : self.hparams.prediction_length, :]
        return preds.squeeze(-1)

    '''The following is the code for the data processing section.'''
    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

