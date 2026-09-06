"""
geometry_spherenet.py — verbatim port of the official SphereNet
`xyz_to_dat` (dig/threedgraph/utils/geometric_computing.py), used ONLY by
`models_3d/spherenet.py`.

REVISION NOTE: this replaces the earlier torsion-index construction
(`geometry_3d.build_torsions`, which picked a 4th atom l bonded to k) --
that was NOT the same geometric quantity the official implementation
computes. The official torsion, for a triplet (k,j,i), is instead
defined via ANOTHER neighbour k_n of the MIDDLE atom j (not of k), and
takes the MINIMUM torsion angle over all such k_n (`scatter(...,
reduce='min')`) as a canonical, choice-independent per-triplet value.
This is a materially different definition, not just a different basis
for the same angle -- hence a verbatim port rather than an adaptation of
the existing torsion machinery. Computed at forward-time directly from
`edge_index`/`pos` (this benchmark's shared radius graph), not cached on
disk -- SphereNet is the only 3D architecture in this benchmark that
needs it, so the shared dataset cache (`tri_idx_*`/`tor_idx_l`/
`tor_has_l` fields used by DimeNet/DimeNet++) is left untouched for
every other architecture.

Requires `torch_scatter` and `torch_sparse` (see requirements-3d.txt) --
dependencies this benchmark previously avoided specifically to sidestep
this function (see the FIDELITY NOTE this replaces in
`models_3d/spherenet.py`'s prior docstring).
"""
import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from math import pi as PI


def xyz_to_dat(pos: torch.Tensor, edge_index: torch.Tensor, num_nodes: int,
               use_torsion: bool = False):
    """Verbatim port of the official function (see module docstring).
    j, i = edge_index (message flows j -> i, matching this benchmark's own
    edge_index convention elsewhere -- edge_index[0]=source/neighbour,
    edge_index[1]=target/center)."""
    j, i = edge_index

    dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

    value = torch.arange(j.size(0), device=j.device)
    adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = i.repeat_interleave(num_triplets)
    idx_j = j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    # Angles, 0 to pi.
    pos_ji = pos[idx_i] - pos[idx_j]
    pos_jk = pos[idx_k] - pos[idx_j]
    a = (pos_ji * pos_jk).sum(dim=-1)
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)
    angle = torch.atan2(b, a)

    if not use_torsion:
        return dist, angle, i, j, idx_kj, idx_ji

    device = pos.device
    idx_batch = torch.arange(len(idx_i), device=device)
    idx_k_n = adj_t[idx_j].storage.col()
    repeat = num_triplets
    num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
    idx_i_t = idx_i.repeat_interleave(num_triplets_t)
    idx_j_t = idx_j.repeat_interleave(num_triplets_t)
    idx_k_t = idx_k.repeat_interleave(num_triplets_t)
    idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
    mask2 = idx_i_t != idx_k_n
    idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = (
        idx_i_t[mask2], idx_j_t[mask2], idx_k_t[mask2], idx_k_n[mask2], idx_batch_t[mask2]
    )

    pos_j0 = pos[idx_k_t] - pos[idx_j_t]
    pos_ji2 = pos[idx_i_t] - pos[idx_j_t]
    pos_jk2 = pos[idx_k_n] - pos[idx_j_t]
    dist_ji = pos_ji2.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(pos_ji2, pos_j0)
    plane2 = torch.cross(pos_ji2, pos_jk2)
    a2 = (plane1 * plane2).sum(dim=-1)
    b2 = (torch.cross(plane1, plane2) * pos_ji2).sum(dim=-1) / dist_ji
    torsion1 = torch.atan2(b2, a2)
    torsion1[torsion1 <= 0] += 2 * PI

    torsion = scatter(torsion1, idx_batch_t, dim=0, dim_size=len(idx_i), reduce="min")
    # NOTE: triplets with NO qualifying k_n (mask2 empties that group) get no
    # entry from scatter's reduce='min' over an empty index group. torch_scatter
    # fills such positions with 0 for "min" reduction (its documented fill
    # value for reductions with no contributing elements), matching the
    # official code's implicit behaviour (it does not special-case this).

    return dist, angle, torsion, i, j, idx_kj, idx_ji
