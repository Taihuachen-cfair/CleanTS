import os
from typing import List, Optional

_DATASET_ROOT = os.path.join("/home/Data1T/chentaihua/Dataset/lotsa_data")  # 可按需修改

# 标准化前缀映射规则
_PREFIX_RULES = [
    ("era5", lambda x: x.lower().startswith("era5")),
    ("cmip6", lambda x: x.lower().startswith("cmip6")),
    ("largest", lambda x: x.lower().startswith("largest")),
]


def _normalize_name(name: str) -> str:
    """将原始文件夹名映射为标准化前缀（如适用），否则保留原名"""
    for prefix, condition in _PREFIX_RULES:
        if condition(name):
            return prefix
    return name


def _load_datasets_from_folder(root: str) -> List[str]:
    if not os.path.isdir(root):
        print(f"Warning: Dataset root directory not found: {root}")
        return []

    raw_subdirs = [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]

    # 标准化名称并去重，同时保留首次出现顺序（用于稳定索引）
    seen = set()
    normalized_list = []
    for name in sorted(raw_subdirs):  # 排序确保跨平台一致
        norm = _normalize_name(name)
        if norm not in seen:
            normalized_list.append(norm)
            seen.add(norm)

    return normalized_list


_DATASETS = _load_datasets_from_folder(_DATASET_ROOT)

# 构建映射：标准化名称 ↔ 索引（1-based）
_FORWARD = {name: idx + 1 for idx, name in enumerate(_DATASETS)}
_BACKWARD = {idx + 1: name for idx, name in enumerate(_DATASETS)}

# 屏蔽名单（标准化后的名称）
_BLOCKLIST = {
    "azure_vm_traces_2017","borg_cluster_data_2011", "residential_pv_power"
}


# 公共接口
def get_dataset_idx(name: str) -> int:
    """输入原始名称或标准化前缀均可"""
    norm_name = _normalize_name(name)
    if norm_name not in _FORWARD:
        valid = list(_FORWARD.keys())
        print(f"Error: Dataset '{name}' (normalized to '{norm_name}') not found. Available: {valid}")
        raise ValueError(f"Dataset '{name}' is not registered.")
    return _FORWARD[norm_name]


def get_dataset_name(idx: int) -> Optional[str]:
    if idx == 0:
        return ""  # 或 None，根据下游需求

    if idx not in _BACKWARD:
        print(f"Error: Dataset index {idx} is invalid. Valid range: 0 to {len(_DATASETS)}")
        raise ValueError(f"Dataset index {idx} out of range.")

    name = _BACKWARD[idx]
    if name in _BLOCKLIST:
        return ""  # 被屏蔽时返回空字符串

    return name