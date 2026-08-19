"""
Model registry for 3D architectures.

Mirrors `src/models/models_2d/__init__.py`'s structure exactly. Each
import is wrapped individually so that a missing OPTIONAL dependency
(currently: only `e3nn`, needed by MACE — see requirements-3d.txt) never
prevents the other 8 architectures from being registered and used; a
warning is logged instead and MACE is simply absent from
`MODEL_REGISTRY_3D` until `pip install -r requirements-3d.txt` is run.
This mirrors the existing top-level `src/models/__init__.py`'s own
try/except around `MODEL_REGISTRY_3D` as a whole.
"""
import logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY_3D = {}

from .schnet import SchNetSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["schnet"] = SchNetSigmaModel

from .painn import PaiNNSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["painn"] = PaiNNSigmaModel

from .dimenet import DimeNetSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["dimenet"] = DimeNetSigmaModel      # pp=False by default in this entry, see configs/3d/dimenet.yaml
MODEL_REGISTRY_3D["dimenet_pp"] = DimeNetSigmaModel   # pp=True, see configs/3d/dimenet_pp.yaml

from .spherenet import SphereNetSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["spherenet"] = SphereNetSigmaModel

from .egnn import EGNNSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["egnn"] = EGNNSigmaModel

from .torchmdnet import TorchMDNetSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["torchmdnet"] = TorchMDNetSigmaModel

from .unimol import UniMolSigmaModel  # noqa: E402
MODEL_REGISTRY_3D["unimol"] = UniMolSigmaModel

try:
    from .mace import MACESigmaModel  # noqa: E402
    MODEL_REGISTRY_3D["mace"] = MACESigmaModel
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    logger.warning(
        f"MACE unavailable ({e}) — install e3nn (`pip install -r "
        f"requirements-3d.txt`) to enable it. All other 3D architectures "
        f"are unaffected."
    )

__all__ = ["MODEL_REGISTRY_3D"]
