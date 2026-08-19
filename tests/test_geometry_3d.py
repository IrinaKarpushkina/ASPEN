"""
test_geometry_3d.py — golden, hand-checkable tests for the pure-PyTorch
geometric preprocessing in `src/data/geometry_3d.py` (radius graph,
DimeNet-style triplets, SphereNet-style torsions), and for
`Data3D.__inc__` (`src/data/dataset_3d.py`), which is the one thing that
makes triplet/torsion indices survive PyG batching correctly (see that
class's docstring). Mirrors the hand-checkable style of
`tests/test_dmpnn_reverse_edge.py` (2D).
"""
import torch
from torch_geometric.data import Batch

from src.data.geometry_3d import radius_graph_single, build_triplets, build_torsions
from src.data.dataset_3d import Data3D


def test_radius_graph_path_of_four_atoms():
    # Linear chain 0-1-2-3, spacing 1.5 A; cutoff 2.0 -> only immediate
    # neighbours connected (1.5 < 2.0 < 3.0).
    pos = torch.tensor([[0., 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]])
    edge_index, edge_weight = radius_graph_single(pos, cutoff=2.0, max_num_neighbors=8)

    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    for j, i, w in zip(edge_index[0].tolist(), edge_index[1].tolist(), edge_weight.tolist()):
        assert abs(w - 1.5) < 1e-5


def test_build_triplets_on_a_bent_three_atom_chain():
    # Bent chain: 0 - 1 - 2 (both directions), atom 1 is the vertex.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    trip = build_triplets(edge_index, num_nodes=3)

    # Edges (index: (src -> dst)): e0=(0->1), e1=(1->0), e2=(1->2), e3=(2->1).
    # incoming[node] = all edges landing on `node`:
    #   incoming[0] = [(1, e1)]
    #   incoming[1] = [(0, e0), (2, e3)]   <- TWO edges land on node 1
    #   incoming[2] = [(1, e2)]
    #
    # For each outer edge (j -> i), pair with every inner edge (k -> j),
    # excluding k == i:
    #   outer e0=(0->1), j=0,i=1: incoming[0]=[(1,e1)] -> k=1==i=1, excluded.
    #   outer e1=(1->0), j=1,i=0: incoming[1]=[(0,e0),(2,e3)] -> k=0==i=0
    #       excluded; k=2!=i=0 KEPT: (k=2, j=1, i=0), idx_kj=e3(=3), idx_ji=e1(=1)
    #   outer e2=(1->2), j=1,i=2: incoming[1]=[(0,e0),(2,e3)] -> k=0!=i=2
    #       KEPT: (k=0, j=1, i=2), idx_kj=e0(=0), idx_ji=e2(=2);
    #       k=2==i=2, excluded.
    #   outer e3=(2->1), j=2,i=1: incoming[2]=[(1,e2)] -> k=1==i=1, excluded.
    assert trip["idx_i"].tolist() == [0, 2]
    assert trip["idx_j"].tolist() == [1, 1]
    assert trip["idx_k"].tolist() == [2, 0]
    assert trip["idx_kj"].tolist() == [3, 0]
    assert trip["idx_ji"].tolist() == [1, 2]


def test_build_torsions_masks_missing_fourth_atom():
    # Same bent 3-atom chain: neither triplet's `k` atom (2, then 0) has
    # any OTHER neighbour besides the triplet's own j/i, so no torsion is
    # available for either triplet.
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    trip = build_triplets(edge_index, num_nodes=3)
    tors = build_torsions(edge_index, trip, num_nodes=3)

    assert tors["has_l"].tolist() == [False, False]
    # dummy fallback must be `i` for each triplet, NOT `k` -- see
    # geometry_3d.py's comment on why `k` would create a degenerate
    # zero-length bond vector. Triplets are (k=2,j=1,i=0) and (k=0,j=1,i=2).
    assert tors["idx_l"].tolist() == [0, 2]


def test_data3d_batching_offsets_triplet_and_torsion_indices_correctly():
    # Two separate small molecules; verify that after batching, EVERY
    # triplet/torsion index still points at nodes/edges belonging to the
    # SAME molecule they came from (no cross-molecule leakage).
    def make(pos, cutoff=2.0):
        n = pos.size(0)
        ei, ew = radius_graph_single(pos, cutoff=cutoff, max_num_neighbors=8)
        trip = build_triplets(ei, n)
        tors = build_torsions(ei, trip, n)
        return Data3D(
            x=torch.zeros(n, 6), z=torch.ones(n, dtype=torch.long),
            pos=pos, y=torch.zeros(n, 51),
            edge_index=ei, edge_weight=ew,
            tri_idx_i=trip["idx_i"], tri_idx_j=trip["idx_j"], tri_idx_k=trip["idx_k"],
            tri_idx_kj=trip["idx_kj"], tri_idx_ji=trip["idx_ji"],
            tor_idx_l=tors["idx_l"], tor_has_l=tors["has_l"],
            num_nodes=n,
        )

    mol_a = make(torch.tensor([[0., 0, 0], [1.5, 0, 0], [3.0, 0, 0]]))       # 3 atoms
    mol_b = make(torch.tensor([[0., 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]]))  # 4 atoms

    batch = Batch.from_data_list([mol_a, mol_b])

    n_a = mol_a.num_nodes
    e_a = mol_a.edge_index.size(1)

    # every triplet/torsion node index belonging to mol_b (which comes
    # second) must be >= n_a (offset applied), and every one belonging to
    # mol_a must be < n_a (NOT accidentally offset).
    ptr = batch.ptr.tolist()  # [0, n_a, n_a+n_b]
    assert ptr == [0, n_a, n_a + mol_b.num_nodes]

    # Reconstruct which triplets came from which molecule via `batch.batch`
    # on the `idx_j` (vertex) node, and check node ids fall in the right range.
    owner = batch.batch[batch.tri_idx_j]
    for owner_id, lo, hi in [(0, 0, n_a), (1, n_a, n_a + mol_b.num_nodes)]:
        sel = owner == owner_id
        for key in ("tri_idx_i", "tri_idx_j", "tri_idx_k"):
            vals = getattr(batch, key)[sel]
            assert (vals >= lo).all() and (vals < hi).all(), (
                f"{key} for molecule {owner_id} leaked outside its own node range "
                f"[{lo}, {hi}) -- Data3D.__inc__ regression"
            )
        for key in ("tri_idx_kj", "tri_idx_ji"):
            vals = getattr(batch, key)[sel]
            elo, ehi = (0, e_a) if owner_id == 0 else (e_a, e_a + mol_b.edge_index.size(1))
            assert (vals >= elo).all() and (vals < ehi).all(), (
                f"{key} for molecule {owner_id} leaked outside its own edge range "
                f"[{elo}, {ehi}) -- Data3D.__inc__ regression"
            )
