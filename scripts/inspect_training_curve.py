"""Print the per-epoch val-loss history from a metrics json - diagnoses
"best_epoch=0" (steadily worsening val from the very start) vs a crash
partway through vs a slow, real plateau.

    python -m scripts.inspect_training_curve results/physics/metrics/sigma_kitchen_sink_seed0_physics_v2.json
"""
import argparse
import json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    a = ap.parse_args()
    r = json.load(open(a.json_path))
    h = r.get("history", [])
    print(f"model={r['model']} loss={r.get('loss')} seed={r.get('seed')} best_epoch={r.get('best_epoch')} "
         f"n_epochs_run={len(h)}")
    if not h:
        print("no 'history' field in this json - can't diagnose from this file")
    else:
        print(f"{'epoch':>6s}{'train_loss':>14s}{'val_loss':>14s}")
        for i, e in enumerate(h):
            mark = " <- best" if i == r.get("best_epoch") else ""
            print(f"{i:>6d}{e.get('train_loss', float('nan')):>14.5f}{e.get('val_loss', float('nan')):>14.5f}{mark}")
    print("\nReading it: if val_loss only goes UP after epoch 0 -> the model itself is unstable at this "
         "lr/schedule (try lower lr, or disable use_bond_graph/use_global one at a time to isolate which "
         "block causes it). If train_loss also stays flat/high -> optimization isn't moving at all (check "
         "for a silently-zero gradient path, e.g. an all-empty bond graph batch combined with a frozen gate).")
