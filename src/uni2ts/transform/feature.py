from dataclasses import dataclass
from typing import Any

import numpy as np
from einops import repeat

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin


@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = False
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.counter = 0
        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )
        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter : self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_seq_id = np.arange(time)
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.observed_mask_field] = observed_mask
        return data_entry

    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)

@dataclass
class AddConstantBoundary(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    fields: tuple[str, ...]
    constant_boundary_field: str = "constant_boundary"
    collection_type: type = list
    eps: float = 1e-5

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        constant_boundary = self.collect_func(
            func=self._generate_constant_boundary,
            data_entry=data_entry,
            fields=self.fields,
        )
        data_entry[self.constant_boundary_field] = constant_boundary
        return data_entry

    def _generate_constant_boundary(self, data_entry: dict[str, Any], field: str, eps: float = 1e-5) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim("target", arr, 2)
        v_dim, t_dim = arr.shape
        patch_size = data_entry["patch_size"]
        boundary_markers = np.zeros_like(arr, dtype=np.int32)

        if t_dim < patch_size:
            return boundary_markers

        # 2. 识别相邻常数关系 (is_constant_pair 形状: [V, T-1])
        # 计算 |x[t+1] - x[t]| <= eps
        is_constant_pair = np.abs(np.diff(arr, axis=1)) <= eps

        # 3. 识别属于“合格长常数段”的点
        # 逻辑：一个点 i 如果属于长度为 K 的常数段，说明它关联的相邻对中必须至少有连续 K-1 个 True。
        # 我们使用滑动窗口求和来快速识别哪些位置满足连续 K-1 个对相等
        k_minus_1 = patch_size - 1

        # 使用卷积/求和查找连续满足条件的对
        # kernel 形状为 [1, K-1]，在时间轴上滑动
        kernel = np.ones(k_minus_1, dtype=np.int32)

        # 为每个变量(行)计算滑动窗口内的 True 个数
        def get_valid_mask(row_pairs):
            # 对每一行计算：是否有连续 k_minus_1 个 True
            consecutive_counts = np.convolve(row_pairs.astype(np.int32), kernel, mode='valid')
            # 只要 count == k_minus_1，说明该起始点开始的 patch_size 个点都是常数
            return consecutive_counts == k_minus_1

        # 初始化长常数掩码 [V, T]
        is_in_long_constant = np.zeros((v_dim, t_dim), dtype=bool)

        for i in range(v_dim):
            has_long_const = get_valid_mask(is_constant_pair[i])
            # 映射回原数组坐标：如果 has_long_const[t] 为 True，
            # 则 arr[i, t:t+patch_size] 这一段都是常数
            for start_idx in np.where(has_long_const)[0]:
                is_in_long_constant[i, start_idx: start_idx + patch_size] = True

        # 4. 识别并标记突变位置 (Transition Points)
        # 计算掩码的变化：1 为进入常数段，-1 为离开常数段
        # 在时间轴两侧补零以便捕获边缘变化
        padded_mask = np.pad(is_in_long_constant, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        diff_mask = np.diff(padded_mask.astype(np.int8), axis=1)  # [V, T+1]

        # starts: diff == 1 (进入), ends: diff == -1 (离开)
        starts_v, starts_t = np.where(diff_mask[:, :-1] == 1)
        ends_v, ends_t = np.where(diff_mask[:, 1:] == -1)

        # 遍历起始点：标记 Transition (Dynamic -> Constant)
        # 过滤掉从序列开头(t=0)就开始的常数段
        valid_start_mask = (starts_t > 0)
        v_idx = starts_v[valid_start_mask]
        t_idx = starts_t[valid_start_mask]
        boundary_markers[v_idx, t_idx - 1] = 1  # 非常数侧
        boundary_markers[v_idx, t_idx] = 1  # 常数侧

        # 遍历结束点：标记 Transition (Constant -> Dynamic)
        # 过滤掉在序列末尾(t=T-1)结束的常数段
        valid_end_mask = (ends_t < t_dim - 1)
        v_idx = ends_v[valid_end_mask]
        t_idx = ends_t[valid_end_mask]
        boundary_markers[v_idx, t_idx] = 1  # 常数侧
        boundary_markers[v_idx, t_idx + 1] = 1  # 非常数侧

        return boundary_markers