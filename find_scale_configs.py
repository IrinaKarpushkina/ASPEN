"""
find_scale_configs.py — подбирает hidden_channels/num_blocks/int_emb_size/
basis_emb_size так, чтобы число параметров модели попало в целевую точку
шкалы (0.5x, 2x, 4x от базы ~719k), и печатает готовые YAML-блоки.

Запускать НА КЛАСТЕРЕ (нужен torch + torch_geometric + сама репа), из корня
ASPEN-benchmark2:

    python find_scale_configs.py --model dimenet_pp_enhanced --targets 0.5 2 4
    python find_scale_configs.py --model dimenet_pp_delta --targets 0.5 2 4 \
        --prior-path /mnt/tank/scratch/ikarpushkina/sigma/ASPEN/data/train_test_val_df/element_sigma_prior.npz

Он не запускает обучение — только считает n_params для разных
hidden_channels при фиксированном num_blocks=3 (масштабируем ширину,
чтобы не увеличивать receptive field и не путать эффект "глубже" с
эффектом "шире"), и подбирает ближайшее значение бинарным поиском.

Кладите этот файл в корень репозитория: ASPEN-benchmark2/find_scale_configs.py
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.physics_extension import (  # noqa: E402
    DimeNetPPEnhanced,
    DimeNetPPDeltaPhysics,
)
from src.models.models_3d.dimenet import DimeNetSigmaModel  # noqa: E402

BASE_N_PARAMS = 719000  # ориентир по вашим таблицам (718872 / 719464)

MODEL_CLS = {
    "dimenet_pp_enhanced": (DimeNetPPEnhanced, dict(input_dim=22)),
    "dimenet_pp_delta": (
        DimeNetPPDeltaPhysics,
        dict(input_dim=22, prior_strength=1.0),  # prior_path добавляется в main() из --prior-path
    ),
    "dimenet_pp": (DimeNetSigmaModel, dict(pp=True)),  # чистый baseline, без input_dim (n_feat берётся из node_feat_dim_3d())
}


def count_params(cls, kwargs, hidden_channels, num_blocks, int_emb_size, basis_emb_size):
    kw = dict(kwargs)
    kw.update(
        hidden_channels=hidden_channels,
        out_emb_channels=hidden_channels,
        num_blocks=num_blocks,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
    )
    m = cls(**kw)
    # prior_logp (у delta) — nn.Buffer, не nn.Parameter, поэтому его размер
    # (87x51, фиксированный, не обучаемый) корректно НЕ входит в счёт ниже.
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def search_hidden(cls, kwargs, target_n, num_blocks=3, int_emb_size=32, basis_emb_size=8,
                   lo=32, hi=512):
    """Бинарный поиск по hidden_channels (кратно 4, т.к. DimeNet++ внутри
    делит hidden на головы/эмбеддинги)."""
    best = None
    while lo <= hi:
        mid = ((lo + hi) // 2) // 4 * 4
        if mid < 4:
            mid = 4
        n = count_params(cls, kwargs, mid, num_blocks, int_emb_size, basis_emb_size)
        if best is None or abs(n - target_n) < abs(best[1] - target_n):
            best = (mid, n)
        if n < target_n:
            lo = mid + 4
        else:
            hi = mid - 4
        if lo > hi:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CLS))
    ap.add_argument("--targets", nargs="+", type=float, default=[0.5, 2, 4],
                     help="множители от базового n_params (~719k)")
    ap.add_argument("--base-n-params", type=float, default=BASE_N_PARAMS)
    ap.add_argument("--prior-path", default=None,
                     help="Путь к element_sigma_prior.npz — обязателен для --model dimenet_pp_delta "
                          "(см. configs/3d/dimenet_pp_delta_fixed.yaml на кластере). Сам прайор — это "
                          "nn.Buffer фиксированного размера (87x51), не обучаемый вес, поэтому на "
                          "число параметров он не влияет, но модель без него не инициализируется.")
    args = ap.parse_args()

    cls, fixed_kwargs = MODEL_CLS[args.model]
    if args.model == "dimenet_pp_delta":
        if not args.prior_path:
            ap.error("--model dimenet_pp_delta требует --prior-path (см. --help)")
        fixed_kwargs = dict(fixed_kwargs, prior_path=args.prior_path)

    print(f"# {args.model}: подбор hidden_channels под целевые множители параметров")
    print(f"# база ~{int(args.base_n_params)} параметров (num_blocks=3, int_emb_size=32, basis_emb_size=8)\n")

    for mult in args.targets:
        target_n = args.base_n_params * mult
        hidden, n = search_hidden(cls, fixed_kwargs, target_n)
        # int_emb_size и basis_emb_size растим пропорционально hidden,
        # иначе на больших hidden они становятся непропорциональным бутылочным горлышком
        int_emb = max(16, round(hidden * 32 / 118 / 4) * 4)
        basis_emb = max(4, round(hidden * 8 / 118 / 2) * 2)
        n_final = count_params(cls, fixed_kwargs, hidden, 3, int_emb, basis_emb)
        ratio = n_final / args.base_n_params
        print(f"## {mult}x (цель {int(target_n)}, получено {n_final}, факт. множитель {ratio:.2f}x)")
        print(f"  model:")
        print(f"    hidden_channels: {hidden}")
        print(f"    out_emb_channels: {hidden}")
        print(f"    int_emb_size: {int_emb}")
        print(f"    basis_emb_size: {basis_emb}")
        print(f"    num_blocks: 3   # фиксируем глубину — масштабируем только ширину")
        print()


if __name__ == "__main__":
    main()
