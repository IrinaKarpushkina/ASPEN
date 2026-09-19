from __future__ import annotations

import numpy as np
import torch

from ..data.dataset_3d import ChaosParquet3DDataset


def build_local_geometry_features(data, z, eneg):
    """Return invariant local-environment features, shape (N, 15).

    Features:
      0-4  neighbour counts within 1.8, 2.2, 2.8, 3.5, 5.0 A
      5-7  radial density sums exp(-d/s) for s=1,2,3 A
      8    sum 1/d
      9    distance-weighted neighbour electronegativity mean
      10   distance-weighted neighbour Z mean
      11   neighbour electronegativity std
      12-14 eigenvalues of the neighbour-direction covariance matrix
    """
    pos = data.pos.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy().astype(np.float32)
    en_np = eneg.detach().cpu().numpy().reshape(-1).astype(np.float32)
    n = len(z_np)
    out = np.zeros((n, 15), dtype=np.float32)

    edge_index = data.edge_index.detach().cpu().numpy()
    edge_weight = data.edge_weight.detach().cpu().numpy()
    nbrs = [[] for _ in range(n)]
    for e in range(edge_index.shape[1]):
        j, i = int(edge_index[0, e]), int(edge_index[1, e])
        if i != j and edge_weight[e] > 1e-6:
            nbrs[i].append((j, float(edge_weight[e])))

    for i in range(n):
        items = nbrs[i]
        if not items:
            continue
        js = np.asarray([x[0] for x in items], dtype=np.int64)
        d = np.asarray([x[1] for x in items], dtype=np.float32)

        for k, r in enumerate((1.8, 2.2, 2.8, 3.5, 5.0)):
            out[i, k] = np.sum(d < r) / 10.0

        out[i, 5] = np.sum(np.exp(-d / 1.0))
        out[i, 6] = np.sum(np.exp(-d / 2.0))
        out[i, 7] = np.sum(np.exp(-d / 3.0))
        out[i, 8] = np.sum(1.0 / np.maximum(d, 0.5)) / 10.0

        w = np.exp(-d / 2.0)
        w /= w.sum() + 1e-8
        out[i, 9] = np.sum(w * en_np[js]) / 4.0
        out[i, 10] = np.sum(w * np.minimum(z_np[js], 86.0)) / 86.0
        out[i, 11] = np.sqrt(np.sum(w * (en_np[js] - np.sum(w * en_np[js])) ** 2))

        vec = pos[js] - pos[i]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        u = vec / np.maximum(norm, 1e-8)
        cov = (u.T * w) @ u
        try:
            eig = np.linalg.eigvalsh(cov)
            out[i, 12:15] = np.sort(eig)
        except np.linalg.LinAlgError:
            pass

    return torch.from_numpy(out)


class EnhancedChaosParquet3DDataset(ChaosParquet3DDataset):
    """Original 3D dataset + invariant local geometry features.

    ``x`` becomes 6 benchmark features + 15 local geometry features = 21.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Re-read only the small mol_id/smiles columns; the expensive 3D
        # geometry remains loaded from the benchmark cache.
        df = __import__("pandas").read_parquet(
            self.parquet_path, columns=["mol_id", "smiles"]
        )
        smiles_by_id = (
            df.drop_duplicates("mol_id").set_index("mol_id")["smiles"].to_dict()
        )

        for data, mol_id in zip(self._data_list, self.mol_ids):
            # Existing x[0] is electronegativity/4, but local features need
            # the unscaled electronegativity. Recover it from z through the
            # benchmark constants instead of storing another tensor.
            from ..data.constants import ELECTRONEGATIVITY, ENEG_DEFAULT
            en = torch.tensor(
                [ELECTRONEGATIVITY.get(int(zi), ENEG_DEFAULT) for zi in data.z],
                dtype=torch.float32,
            )
            local = build_local_geometry_features(data, data.z, en)

            # Add element-only polarizability as a cheap electronic proxy.
            from ..data.constants import POLARIZABILITY, POLARIZABILITY_DEFAULT
            pol = torch.tensor(
                [[POLARIZABILITY.get(int(zi), POLARIZABILITY_DEFAULT) / 6.0]
                 for zi in data.z],
                dtype=torch.float32,
            )
            data.x = torch.cat([data.x.float(), pol, local], dim=1)

        self.enhanced_feature_dim = int(self._data_list[0].x.shape[1])
