from .layers import ResidualMLP
from .gcn import GCNSigmaModel
from .gat import GATSigmaModel
from .gatv2 import GATv2SigmaModel
from .gine import GINESigmaModel
from .schnet import SchNetSigmaModel
from .final_model import FinalSigmaModel
from .attentive_fp import AtomAttentiveFPModel
from .dmpnn import DMPNNSigmaModel
from .gps import GPSSigmaModel


MODEL_REGISTRY = {
    "gcn":   GCNSigmaModel,
    "gat":   GATSigmaModel,
    "gatv2": GATv2SigmaModel,
    "gine":  GINESigmaModel,
    "schnet": SchNetSigmaModel,
    "final": FinalSigmaModel,
    "attentive_fp": AtomAttentiveFPModel,
    "dmpnn":        DMPNNSigmaModel,
    "gps":          GPSSigmaModel,
}
