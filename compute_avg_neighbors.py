import sys
sys.path.insert(0, ".")

from src.data.dataset_3d import ChaosParquet3DDataset

ds = ChaosParquet3DDataset(
    "data/train_test_val_df/chaos_atomic_train_with_coordinates.parquet",
    # оставляем дефолты: require_provided_coords=True, compute_triplets=True
)

total_edges, total_nodes = 0, 0
n = min(2000, len(ds))
for i in range(n):
    d = ds[i]
    total_edges += d.edge_index.shape[1]
    total_nodes += d.num_nodes

avg = total_edges / total_nodes
print(f"avg_num_neighbors: {avg:.4f}")
print(f"Checked molecules: {n} / {len(ds)}")
