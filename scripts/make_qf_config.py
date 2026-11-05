"""Создаёт configs/3d/sigma_dn_geom_qf.yaml из ВАШЕГО configs/3d/sigma_dn_geom.yaml: копия один в один,
меняются только model.name и тип головы (head: quantile + её параметры). Так абляция чистая:
backbone, признаки, loss, обучение, defaults - буквально те же. Запуск из корня проекта."""
import os, sys
import yaml

SRC, DST = "configs/3d/sigma_dn_geom.yaml", "configs/3d/sigma_dn_geom_qf.yaml"
cfg = yaml.safe_load(open(SRC))
m = cfg["model"]
if m.get("name") != "sigma_dn_geom":
    sys.exit(f"в {SRC} model.name = {m.get('name')!r}, ожидалось 'sigma_dn_geom'")
if m.get("neutral"):
    sys.exit("neutral: true не поддерживается квантильной головой")
m["name"] = "sigma_dn_geom_qf"
m["head"] = "quantile"
m["n_knots"] = 64          # число отрезков квантильной функции
m["tau_kind"] = "cos"      # сетка уровней tau сгущена к 0 и 1 (хвосты = полярные бины)
m["gap_eps"] = 1.0e-4      # минимальный зазор между узлами (доля диапазона sigma)
if os.path.exists(DST):
    print("уже есть, перезаписываю:", DST)
with open(DST, "w") as f:
    f.write("# Сгенерировано scripts/make_qf_config.py из sigma_dn_geom.yaml.\n"
            "# Отличие от sigma_dn_geom ТОЛЬКО в голове: квантильная функция вместо softmax по бинам + проекции момента.\n")
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print("создан:", DST)
for k in ("name", "head", "n_knots", "tau_kind", "gap_eps", "use_geom_area", "use_global", "input_dim"):
    print(f"  model.{k} = {m.get(k)}")
print("  features =", cfg.get("features"), "| loss =", cfg.get("loss"))
