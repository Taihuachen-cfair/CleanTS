from dataclasses import dataclass
from typing import Any

import numpy as np

from ._base import Transformation
from ..common.dataset_registry import get_dataset_idx


class DebugShape(Transformation):
    def __init__(self, name: str = ""):
        self.name = name

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        print(f"==[DebugShape] {self.name}==")
        self._debug_item(data_entry, indent=0)
        # item_id = data_entry.get("item_id")
        # if item_id =="19174":
        #     print(f"==[DebugShape] {self.name}== (item_id={item_id})")
        #     self._debug_item(data_entry, indent=0)
        return data_entry

    def _debug_item(self, item: Any, key: str = "", indent: int = 0):
        prefix = "  " * indent
        if isinstance(item, dict):
            if key:
                print(f"{prefix}{key}: dict")
            for subkey, subval in item.items():
                self._debug_item(subval, key=subkey, indent=indent + 1)
        elif isinstance(item, (list, tuple)):
            print(f"{prefix}{key}: list/tuple, len={len(item)}")
        elif hasattr(item, "shape") and not np.isscalar(item):
            print(f"{prefix}{key}: shape={tuple(item.shape)}")
        elif isinstance(item, (int, float, str, np.integer, np.floating)):
            print(f"{prefix}{key}: value={item}")
        else:
            print(f"{prefix}{key}: type={type(item)}")

@dataclass
class RecordDataset(Transformation):
    dataset_name: str
    dataset_field: str = "dataset"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry[self.dataset_field] = np.int_(get_dataset_idx(self.dataset_name))
        return data_entry