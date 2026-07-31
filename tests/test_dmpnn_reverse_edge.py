"""
test_dmpnn_reverse_edge.py — golden test for the D-MPNN "reverse-edge
masking" trick (Yang et al., 2019; see src/models/dmpnn.py docstring),
which is the one property that makes D-MPNN different from a plain
edge-centric MPNN: when aggregating incoming messages for edge v->w, the
message coming from the reverse edge w->v must be EXCLUDED.

This test builds a tiny hand-checkable graph and verifies:
  1. build_reverse_index correctly pairs each directed edge with its
     reverse.
  2. DMPNNConv's aggregation for edge (v->w) equals
     sum_{k->v, k != w} h_{kv}, NOT sum_{k->v} h_{kv} (i.e. the reverse
     edge is actually excluded, not just present in the code as a comment).
"""
import torch

from src.models.models_2d.dmpnn import DMPNNConv, build_reverse_index


def test_build_reverse_index_pairs_edges_correctly():
    # Path graph 0 - 1 - 2, both directions: (0,1),(1,0),(1,2),(2,1)
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ], dtype=torch.long)
    rev = build_reverse_index(edge_index)

    # edge 0: (0->1), reverse should be edge 1: (1->0)
    # edge 1: (1->0), reverse should be edge 0: (0->1)
    # edge 2: (1->2), reverse should be edge 3: (2->1)
    # edge 3: (2->1), reverse should be edge 2: (1->2)
    assert rev.tolist() == [1, 0, 3, 2]


def test_reverse_edge_is_actually_excluded_from_aggregation():
    torch.manual_seed(0)
    hidden = 8
    # Star graph: node 0 is the center, connected to 1, 2, 3 (both directions).
    # Edges, directed: 1->0, 2->0, 3->0, 0->1, 0->2, 0->3
    edge_index = torch.tensor([
        [1, 2, 3, 0, 0, 0],
        [0, 0, 0, 1, 2, 3],
    ], dtype=torch.long)
    num_nodes = 4
    rev_idx = build_reverse_index(edge_index)

    conv = DMPNNConv(hidden, dropout=0.0)
    conv.eval()

    h_edge = torch.randn(edge_index.size(1), hidden)
    h_init = torch.randn_like(h_edge)

    with torch.no_grad():
        out = conv(h_edge, h_init, edge_index, rev_idx, num_nodes)

    # Manually recompute agg_edge for edge e=0, which is (1->0):
    # agg_full[1] = sum of h_edge over edges with target==1 = h_edge[3] (0->1)
    # agg_{1\0} should EXCLUDE the reverse edge (0->1) itself... but here the
    # reverse of edge (1->0) IS edge (0->1) = edge index 3, which is also the
    # only incoming edge to node 1. So agg_{1\0} must be exactly zero.
    agg_full_node1 = h_edge[3]  # only incoming edge to node 1 is edge index 3 (0->1)
    reverse_of_edge0 = h_edge[rev_idx[0]]
    agg_edge0_manual = agg_full_node1 - reverse_of_edge0
    assert torch.allclose(agg_edge0_manual, torch.zeros(hidden), atol=1e-6), (
        "Test setup assumption violated: expected the only incoming message "
        "to node 1 to be exactly the reverse edge of edge 0."
    )

    expected_out0 = conv.act(conv.norm(conv.W_i(h_init[0]) + conv.W_m(agg_edge0_manual)))
    assert torch.allclose(out[0], expected_out0, atol=1e-5), (
        "DMPNNConv is not correctly excluding the reverse edge from the "
        "aggregation for a star-graph leaf edge — this is the defining "
        "correctness property of D-MPNN (Yang et al., 2019)."
    )
