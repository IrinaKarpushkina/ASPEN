"""Enhanced (22-dim) dataset + 3 SASA features (25-dim) + ``data.sasa_ref``.

Use with ``features: sasa`` in the yaml (see scripts/patch_train_3d_physics.py).
SASA is computed once and cached next to the graph cache.
"""
from __future__ import annotations

import logging
import os

import torch

from .dataset_3d_enhanced import EnhancedChaosParquet3DDataset
from .features_3d_sasa import PROBES, SASA_INPUT_SCALE, sasa_features_np

logger = logging.getLogger(__name__)
SASA_CACHE_VERSION = "sasa_v1"


class SasaChaosParquet3DDataset(EnhancedChaosParquet3DDataset):
    def __init__(self, parquet_path, *args, ref_probe_index: int = 0, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        force = bool(kwargs.get("force_recompute", False))
        super().__init__(parquet_path, *args, **kwargs)

        cache_dir = cache_dir or os.path.join(os.path.dirname(parquet_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{os.path.basename(parquet_path)}.{SASA_CACHE_VERSION}.pt")

        feats = None
        if os.path.exists(path) and not force:
            try:
                feats = torch.load(path, weights_only=False)
                ok = len(feats) == len(self._data_list) and all(
                    f.shape == (d.num_nodes, len(PROBES)) for f, d in zip(feats, self._data_list))
                if not ok:
                    logger.warning("SASA cache does not match dataset, recomputing")
                    feats = None
            except Exception as e:  # corrupt cache
                logger.warning("Could not read SASA cache (%s), recomputing", e)
                feats = None

        if feats is None:
            logger.info("Computing SASA for %d molecules (one-off, cached to %s)", len(self._data_list), path)
            feats = []
            for k, d in enumerate(self._data_list):
                f = sasa_features_np(d.pos.cpu().numpy(), d.z.cpu().numpy())
                feats.append(torch.from_numpy(f))
                if (k + 1) % 5000 == 0:
                    logger.info("  SASA %d/%d", k + 1, len(self._data_list))
            torch.save(feats, path)

        for d, f in zip(self._data_list, feats):
            d.x = torch.cat([d.x, f / SASA_INPUT_SCALE], dim=1)
            d.sasa_ref = f[:, ref_probe_index].clone()
        self.sasa_feature_dim = len(PROBES)
        logger.info("SASA features added: x dim -> %d", self._data_list[0].x.shape[1])
