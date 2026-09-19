"""Thin enhanced-feature wrapper around the benchmark 3D dataset."""
from __future__ import annotations

import logging

from .dataset_3d import ChaosParquet3DDataset
from .features_3d_enhanced import augment_3d_features, ENHANCED_FEATURE_DIM

logger = logging.getLogger(__name__)


class EnhancedChaosParquet3DDataset(ChaosParquet3DDataset):
    """Original cached 3D dataset with deterministic invariant feature augmentation.

    The expensive graph/coordinate construction is still performed exactly once
    by ``ChaosParquet3DDataset``.  Enhancement happens only in memory after the
    base cache has been loaded, so the benchmark's graph construction and split
    are unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for data in self._data_list:
            data.x = augment_3d_features(data)
        self.enhanced_feature_dim = ENHANCED_FEATURE_DIM
        logger.info(
            "Enhanced 3D features enabled: %d -> %d dimensions",
            6,
            self.enhanced_feature_dim,
        )
