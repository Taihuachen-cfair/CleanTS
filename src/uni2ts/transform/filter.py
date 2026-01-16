from dataclasses import dataclass
from typing import Any, Optional

from ._base import Transformation

@dataclass
class FilterByLength(Transformation):
    field: str
    min_length: int

    def __call__(self, data_entry: dict[str, Any]) -> Optional[dict[str, Any]]:
        if len(data_entry[self.field]) >= self.min_length:
            return data_entry
        return None
