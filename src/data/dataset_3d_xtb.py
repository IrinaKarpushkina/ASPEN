"""Датасеты с xTB-признаками.

features: xtb       = enhanced(22) + xTB(7)            = 29 dim   (для dimenet_pp_enhanced_xtb / delta_xtb)
features: xtb_sasa  = enhanced(22) + SASA(3) + xTB(7) = 32 dim + data.sasa_ref (для sigma_*)

Граф, сплит и кэш базовых классов не меняются: xTB-массивы читаются из отдельного .npz
(scripts/xtb_features.py) и присоединяются в памяти.
"""
from __future__ import annotations

from .dataset_3d_enhanced import EnhancedChaosParquet3DDataset
from .dataset_3d_sasa import SasaChaosParquet3DDataset
from .features_xtb import XTB_NODE_DIM, attach_xtb


class XtbChaosParquet3DDataset(EnhancedChaosParquet3DDataset):
    def __init__(self, parquet_path, *args, **kwargs):
        super().__init__(parquet_path, *args, **kwargs)
        attach_xtb(self._data_list, parquet_path)
        self.xtb_feature_dim = XTB_NODE_DIM


class XtbSasaChaosParquet3DDataset(SasaChaosParquet3DDataset):
    def __init__(self, parquet_path, *args, **kwargs):
        super().__init__(parquet_path, *args, **kwargs)
        attach_xtb(self._data_list, parquet_path)
        self.xtb_feature_dim = XTB_NODE_DIM
