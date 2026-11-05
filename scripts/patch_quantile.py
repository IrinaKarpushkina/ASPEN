"""Подключает квантильную голову к src/train_3d_physics.py (идемпотентно, с бэкапом .bak_qf).
Модель с `head: quantile` в блоке model yaml строится как DimeNetPPSigmaQuantile.
Запуск из корня проекта: python scripts/patch_quantile.py"""
import os, shutil, sys

P = "src/train_3d_physics.py"
ANCHOR = '    if name.startswith("sigma_surface"):'
MARK = "# [qf-patch]"
INSERT = '''    if mc.pop("head", "softmax") == "quantile":  # [qf-patch]
        from .physics_extension.sigma_quantile import DimeNetPPSigmaQuantile
        return DimeNetPPSigmaQuantile(**mc), name
'''
s = open(P).read()
if MARK in s:
    print("уже пропатчено:", P); sys.exit(0)
if s.count(ANCHOR) != 1:
    sys.exit(f"не найдено место вставки ({ANCHOR!r}) в {P}: пришлите мне функцию build_model")
shutil.copy(P, P + ".bak_qf")
open(P, "w").write(s.replace(ANCHOR, INSERT + ANCHOR))
print("пропатчено:", P, "(бэкап", P + ".bak_qf)")
