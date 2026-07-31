"""
test_gcn_conv_formula.py — golden test comparing torch_geometric's GCNConv
against the propagation rule from the original paper (Kipf & Welling,
ICLR 2017, eq. 2):

    H^{(l+1)} = sigma( D~^{-1/2} A~ D~^{-1/2} H^{(l)} W^{(l)} )

where A~ = A + I (self-loops) and D~ is the degree matrix of A~.

This guards against silent normalization bugs (wrong side of D^{-1/2},
missing self-loops, etc.) in the layer this benchmark relies on for the
GCN architecture — since GCNConv is imported from torch_geometric rather
than reimplemented, this test also serves as a "is our installed PyG
version still doing what the paper says" regression check.
"""
import torch
from torch_geometric.nn import GCNConv


def test_gcnconv_matches_normalized_adjacency_propagation():
    torch.manual_seed(0)

    # Tiny 4-node ring graph: 0-1-2-3-0
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
    ], dtype=torch.long)
    N, in_dim, out_dim = 4, 5, 3
    x = torch.randn(N, in_dim)

    conv = GCNConv(in_dim, out_dim, bias=False, add_self_loops=True, normalize=True)
    with torch.no_grad():
        out = conv(x, edge_index)

    # Hand-build A (with self-loops) and D^{-1/2} A D^{-1/2}, per eq. 2.
    A = torch.zeros(N, N)
    for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        A[i, j] = 1.0
    A = A + torch.eye(N)  # self-loops
    deg = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # GCNConv applies: A_norm @ (x @ W^T)   (no activation, no bias here)
    W = conv.lin.weight  # (out_dim, in_dim)
    expected = A_norm @ (x @ W.T)

    assert torch.allclose(out, expected, atol=1e-5), (
        "GCNConv output does not match the D^{-1/2} A~ D^{-1/2} X W propagation "
        "rule from Kipf & Welling (2017, eq. 2). Either the installed "
        "torch_geometric version changed GCNConv's normalization, or "
        "add_self_loops/normalize flags were changed from the paper-faithful "
        "settings used in src/models/gcn.py."
    )
