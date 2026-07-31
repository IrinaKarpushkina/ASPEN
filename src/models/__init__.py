from .models_2d import MODEL_REGISTRY_2D

try:
    from .models_3d import MODEL_REGISTRY_3D
except (ImportError, ModuleNotFoundError):
    MODEL_REGISTRY_3D = {}

# Единый реестр всех моделей (2D + 3D)
MODEL_REGISTRY = {
    **MODEL_REGISTRY_2D,
    **MODEL_REGISTRY_3D,
}

__all__ = ["MODEL_REGISTRY", "MODEL_REGISTRY_2D", "MODEL_REGISTRY_3D"]
