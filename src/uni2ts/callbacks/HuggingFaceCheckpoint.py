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

import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set, Union
from weakref import proxy

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import (
    WarningCache,
    rank_zero_info,
    rank_zero_warn,
)

log = logging.getLogger(__name__)
warning_cache = WarningCache()

_PATH = Union[str, Path]


class HuggingFaceCheckpoint(ModelCheckpoint):
    r"""
    Save the model to the Hugging Face format.
    Inherit from `pytorch_lightning.callbacks.ModelCheckpoint` and override the `_save_checkpoint` method.
    """

    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        save_top_k: int = 1,
        mode: str = "min",
        every_n_epochs: Optional[int] = None,
        every_n_train_steps: Optional[int] = None,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            every_n_epochs=every_n_epochs,
            every_n_train_steps=every_n_train_steps
        )

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # Only save the checkpoint if it is in the main process
        if not trainer.is_global_zero:
            return

        # Extract the actual model for saving (strip Lightning and DDP wrappers)
        pl_module = trainer.model

        # Step 1: Strip LightningModule wrapper if needed
        # Some setups may wrap model inside .model attribute
        model_to_save = getattr(pl_module, "model", pl_module)

        # Step 2: Strip DDP/DataParallel wrapper if exists
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # Now model_to_save should be your HuggingFace-style model with save_pretrained
        # filepath in pytorch lightning usually ends with .ckpt
        # To get the directory to save the model, remove the .ckpt
        save_dir = filepath[:-5] if filepath.endswith(".ckpt") else filepath

        try:
            model_to_save.save_pretrained(save_dir)
            self._last_global_step_saved = trainer.global_step
            self._last_checkpoint_saved = save_dir

            # notify loggers
            if trainer.is_global_zero:
                for logger in trainer.loggers:
                    logger.after_save_checkpoint(proxy(self))

        except Exception as e:
            warnings.warn(f"An error occurred during model saving: {e}")

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        # Only remove the checkpoint if it is in the main process
        if not trainer.is_global_zero:
            return

        # filepath in pytorch lightning usually ends with .ckpt
        # To get the directory to save the model, remove the .ckpt
        if filepath.endswith(".ckpt"):
            save_dir = filepath.split(".ckpt")[0]
        else:
            save_dir = filepath
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        else:
            warnings.warn(f"Checkpoint not found: {save_dir}")
