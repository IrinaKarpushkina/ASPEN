"""
Model registry for 2D architectures.
"""
from ..layers import ResidualMLP
from .gcn import GCNSigmaModel
from .gat import GATSigmaModel
from .gatv2 import GATv2SigmaModel
from .gine import GINESigmaModel
from .dmpnn import DMPNNSigmaModel
from .attentive_fp import AtomAttentiveFPModel
from .gps import GPSSigmaModel

MODEL_REGISTRY_2D = {
    "gcn": GCNSigmaModel,
    "gat": GATSigmaModel,
    "gatv2": GATv2SigmaModel,
    "gine": GINESigmaModel,
    "dmpnn": DMPNNSigmaModel,
    "attentive_fp": AtomAttentiveFPModel,
    "gps": GPSSigmaModel,
}
