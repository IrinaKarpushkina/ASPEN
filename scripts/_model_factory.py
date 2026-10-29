"""Сборка (модель, Dataset-класс, kwargs) по конфигу для base-моделей (MODEL_REGISTRY) и physics/xtb/surface-моделей.
Логика выбора повторяет src/train_3d.py и src/train_3d_physics.py."""
from __future__ import annotations


def build_for_eval(cfg: dict, mode: str):
    from src.models import MODEL_REGISTRY
    data_cfg, name = cfg["data"], cfg["model"]["name"]
    mc = dict(cfg["model"]); mc.pop("name", None); mc.pop("n_params_target", None)
    if mode == "2d":
        from src.data.dataset import ChaosParquetDataset
        return MODEL_REGISTRY[name](**mc), ChaosParquetDataset, dict(cache_dir=data_cfg.get("cache_dir"))
    if name in MODEL_REGISTRY:
        from src.data.dataset_3d import ChaosParquet3DDataset
        trip = name in {"dimenet", "dimenet_pp", "spherenet"}
        return MODEL_REGISTRY[name](**mc), ChaosParquet3DDataset, dict(cache_dir=data_cfg.get("cache_dir"), compute_triplets=trip)
    import src.train_3d_physics as T
    from src.data.dataset_3d import ChaosParquet3DDataset
    from src.data.dataset_3d_enhanced import EnhancedChaosParquet3DDataset
    from src.data.dataset_3d_sasa import SasaChaosParquet3DDataset
    from src.data.dataset_3d_kitchen import KitchenChaosParquet3DDataset
    classes = {"base": ChaosParquet3DDataset, "enhanced": EnhancedChaosParquet3DDataset,
               "sasa": SasaChaosParquet3DDataset, "kitchen": KitchenChaosParquet3DDataset}
    try:
        from src.data.dataset_3d_xtb import XtbChaosParquet3DDataset, XtbSasaChaosParquet3DDataset
        classes.update({"xtb": XtbChaosParquet3DDataset, "xtb_sasa": XtbSasaChaosParquet3DDataset})
    except Exception:
        pass
    trip = (name in T._MODELS_NEEDING_TRIPLETS or name.startswith("sigma_dn")
            or (name == "sigma_kitchen_sink" and cfg["model"].get("backbone", "dimenet") == "dimenet")
            or name == "sigma_continuous_dimenet" or name.startswith("sigma_surface"))
    model, _ = T.build_model(cfg)
    return model, classes[cfg.get("features", "base")], dict(
        cache_dir=data_cfg.get("cache_dir"), compute_triplets=trip, require_provided_coords=True)
