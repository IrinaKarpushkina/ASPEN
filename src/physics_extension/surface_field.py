"""Surface neural field: sigma-profile атома как дифференцируемая гистограмма по точкам поверхности.

Физика цели. Атомный sigma-профиль = (площадь поверхности полости атома) x (распределение значений
плотности экранирующего заряда sigma на этой поверхности). Поэтому вместо "вектор -> 51 бин" модель
делает то, что делает сам расчёт:

  1. вокруг атома i строит P точек на сфере радиуса R_i = 1.17 * vdW (как ваш SASA-код) и находит,
     какие из них открыты (не внутри сфер соседей) -> e_ip in {0,1};
  2. для каждой точки считает ЛОКАЛЬНЫЕ ПОЛЕВЫЕ ПРИЗНАКИ без ограничения cutoff 5 A:
     потенциал phi, нормальную E_n и тангенциальную E_t компоненты поля от зарядов xTB ВСЕХ
     остальных атомов молекулы, расстояние до поверхности ближайшего соседа, косинусы угла между
     направлением на точку и связями (c_back = ось sigma-hole);
  3. точечная MLP по (h_i, признаки точки) выдаёт для каждой точки гауссиану: центр mu_p, ширину s_p
     и лог-вес g_p. Гауссианы нормированы по 51 бину, веса w_p ~ e_p * exp(g_p);
  4. профиль = площадь_i * sum_p w_p * Gauss_p(sigma_b) / d_sigma, площадь = SASA * exp(поправка)
     (та же площадная голова, что в SigmaGeomHead, поэтому calibrate() идентична).

Положительность и согласованность площади выполняются по построению. Ограничение: у вас НЕТ sigma в
точках, только гистограммы, поэтому обучение идёт по гистограмме (loss тот же), а mu_p/s_p
неидентифицируемы по отдельности - это смесь, а не "истинное поле".
Поверхностная сетка - в лабораторной системе координат, поэтому при обучении она случайно
поворачивается на каждый батч (rotation augmentation), при валидации/тесте фиксирована.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from ..data.constants import DELTA_SIGMA, MAX_Z, VDW_DEFAULT, VDW_RADII
from .models import DimeNetPPExperimentBackbone
from .physics import SIGMA, SIGMA_MAX

logger = logging.getLogger(__name__)

ANG2BOHR = 1.0 / 0.52917721067
N_PT_FEATS = 7


def _vdw_lut() -> torch.Tensor:
    lut = torch.full((MAX_Z + 1,), float(VDW_DEFAULT))
    for z, r in VDW_RADII.items():
        if z <= MAX_Z:
            lut[z] = float(r)
    return lut


def fibonacci_sphere(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    u = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=1)
    return u.to(dtype)


def random_rotation(device=None, dtype=torch.float32) -> torch.Tensor:
    a = torch.randn(3, 3, device=device, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(dtype)


@torch.no_grad()
def surface_point_features(pos, z, q, batch, n_points: int = 48, radius_scale: float = 1.17,
                           use_field: bool = True, rot=None, max_elems: float = 4e7):
    """Обёртка: режет батч на подряд идущие группы молекул так, чтобы (B*Nmax^2*P) <= max_elems
    (защита от OOM, если в батче попалась очень большая молекула). Результат не зависит от разбиения."""
    counts = torch.bincount(batch).tolist()
    if len(counts) == 1 or len(counts) * max(counts) ** 2 * n_points <= max_elems:
        return _surface_core(pos, z, q, batch, n_points, radius_scale, use_field, rot)
    feats, expo, start, k = [], [], 0, 0
    while k < len(counts):
        m, nmax, tot = 0, 0, 0
        while k + m < len(counts):
            n2 = max(nmax, counts[k + m])
            if m > 0 and (m + 1) * n2 ** 2 * n_points > max_elems:
                break
            nmax, tot, m = n2, tot + counts[k + m], m + 1
        sl = slice(start, start + tot)
        f, e = _surface_core(pos[sl], z[sl], q[sl], batch[sl] - batch[sl][0], n_points, radius_scale, use_field, rot)
        feats.append(f)
        expo.append(e)
        start, k = start + tot, k + m
    return torch.cat(feats), torch.cat(expo)


@torch.no_grad()
def _surface_core(pos: torch.Tensor, z: torch.Tensor, q: torch.Tensor, batch: torch.Tensor,
                  n_points: int = 48, radius_scale: float = 1.17, use_field: bool = True,
                  rot: torch.Tensor | None = None):
    """Возвращает feats (N, P, N_PT_FEATS) и expo (N, P) (1 = точка открыта).

    Всё считается плотно по молекуле (без cutoff): (B, Nmax, P, Nmax) float32 - для батча из 24 молекул по
    ~100 атомов это ~11 млн элементов на тензор. Всё без градиента.
    rot: матрица поворота сферы (P, 3) @ rot.T; если None - каноническая сетка.
    """
    dev = pos.device
    with torch.autocast(device_type=dev.type, enabled=False):
        pos = pos.float()
        xd, mask = to_dense_batch(pos, batch)                     # (B, Nm, 3), (B, Nm)
        zd, _ = to_dense_batch(z.long().clamp(0, MAX_Z), batch)
        qd, _ = to_dense_batch(q.float(), batch)
        B, Nm, _ = xd.shape
        P = n_points
        U = fibonacci_sphere(P, dev)
        if rot is not None:
            U = U @ rot.to(dev).T
        R = _vdw_lut().to(dev)[zd] * radius_scale                  # (B, Nm)
        pts = xd[:, :, None, :] + R[:, :, None, None] * U[None, None]          # (B, Nm, P, 3)
        d = torch.cdist(pts.reshape(B, Nm * P, 3), xd,
                        compute_mode="donot_use_mm_for_euclid_dist").reshape(B, Nm, P, Nm)
        eye = torch.eye(Nm, dtype=torch.bool, device=dev)[None, :, None, :]
        valid_j = mask[:, None, None, :] & ~eye                    # (B, Nm, 1, Nm)
        covered = ((d < R[:, None, None, :]) & valid_j).any(-1)    # (B, Nm, P)
        expo = (~covered).float()
        dnn = (d - R[:, None, None, :]).masked_fill(~valid_j, 1e3).min(-1).values.clamp(0.0, 3.0)

        if use_field:
            db = (d * ANG2BOHR).clamp_min(1.0)
            qj = (qd * mask)[:, None, None, :] * valid_j
            inv = 1.0 / db
            phi = (qj * inv).sum(-1)                               # (B, Nm, P)  [a.u.]
            W = qj * inv ** 3
            E = ANG2BOHR * (pts * W.sum(-1, keepdim=True) - torch.einsum("bipj,bjc->bipc", W, xd))
            En = (E * U[None, None]).sum(-1)
            Et = (E - En.unsqueeze(-1) * U[None, None]).norm(dim=-1)
            f_phi, f_en, f_et = (torch.asinh(phi / 0.02), torch.asinh(En / 0.02), torch.asinh(Et / 0.02))
            del W, E
        else:
            f_phi = f_en = f_et = torch.zeros_like(dnn)
        del d

        dij = torch.cdist(xd, xd, compute_mode="donot_use_mm_for_euclid_dist")      # (B, Nm, Nm)
        vdw = _vdw_lut().to(dev)[zd]
        pair_ok = mask[:, :, None] & mask[:, None, :] & ~torch.eye(Nm, dtype=torch.bool, device=dev)[None]
        bonded = (dij < 0.62 * (vdw[:, :, None] + vdw[:, None, :])) & pair_ok       # (B, Nm, Nm)
        S = torch.einsum("bic,pc->bip", xd, U)                                       # (B, Nm, P)
        cosv = (S.permute(0, 2, 1)[:, None] - S[:, :, :, None]) / dij.clamp_min(1e-6)[:, :, None, :]
        bm = bonded[:, :, None, :]
        c_fwd = cosv.masked_fill(~bm, -1.0).max(-1).values
        c_back = (-cosv).masked_fill(~bm, -1.0).max(-1).values
        has_b = bonded.any(-1)[:, :, None]
        c_fwd = torch.where(has_b, c_fwd, torch.zeros_like(c_fwd))
        c_back = torch.where(has_b, c_back, torch.zeros_like(c_back))

        feats = torch.stack([f_phi, f_en, f_et, dnn / 2.0, expo, c_fwd, c_back], dim=-1)   # (B, Nm, P, 7)
        return feats[mask], expo[mask]


class SigmaSurfaceField(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_channels: int = 118, dropout: float = 0.05,
                 n_points: int = 48, point_hidden: int = 96, use_xtb_field: bool = True,
                 use_geom_area: bool = True, radius_scale: float = 1.17,
                 min_width: float = 0.6, g_scale: float = 1.0, rotate_train: bool = True, **kwargs):
        super().__init__()
        self.encoder = DimeNetPPExperimentBackbone(input_dim, hidden_channels, dropout, **kwargs)
        H = hidden_channels
        self.n_points, self.radius_scale = int(n_points), float(radius_scale)
        self.use_xtb_field, self.rotate_train = bool(use_xtb_field), bool(rotate_train)
        self.min_width, self.g_scale = float(min_width), float(g_scale)
        self.atom_proj = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, point_hidden))
        self.point_in = nn.Linear(N_PT_FEATS, point_hidden)
        self.point_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(point_hidden, point_hidden),
            nn.SiLU(), nn.Linear(point_hidden, point_hidden),
            nn.SiLU(), nn.Linear(point_hidden, 3))
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)
        self.register_buffer("width_bias", torch.tensor(math.log(math.expm1(1.5))))
        self.area = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H // 2), nn.SiLU(), nn.Linear(H // 2, 1))
        nn.init.zeros_(self.area[-1].weight)
        nn.init.zeros_(self.area[-1].bias)
        self.use_geom_area = use_geom_area
        self.log_area_bias = nn.Parameter(torch.zeros(()))
        self.log_area_scale = nn.Parameter(torch.tensor(1.0 if use_geom_area else 0.0), requires_grad=use_geom_area)

    @torch.no_grad()
    def calibrate(self, dataset):
        vals = []
        for d in dataset._data_list:
            a = d.y.clamp_min(0).sum(dim=1) * DELTA_SIGMA
            ok = a > 1e-6
            lr = torch.log(a.clamp_min(1e-8))
            if self.use_geom_area:
                if not hasattr(d, "sasa_ref"):
                    raise RuntimeError("sigma_surface: нужен features: xtb_sasa или sasa (нет data.sasa_ref)")
                lr = lr - torch.log(d.sasa_ref.clamp_min(1e-3))
            vals.append(lr[ok])
        b = torch.cat(vals).median()
        self.log_area_bias.fill_(float(b))
        logger.info("calibrate: log_area_bias = %.4f", float(b))

    def forward(self, data):
        h = self.encoder(data)
        dev = h.device
        with torch.autocast(device_type=dev.type, enabled=False):
            h32 = h.float()
            q = getattr(data, "xtb_q", None)
            if q is None:
                q = torch.zeros(h32.shape[0], device=dev)
                use_field = False
            else:
                use_field = self.use_xtb_field
            rot = random_rotation(dev) if (self.training and self.rotate_train) else None
            feats, expo = surface_point_features(data.pos, data.z, q, data.batch, self.n_points,
                                                 self.radius_scale, use_field, rot)
            zz = self.atom_proj(h32).unsqueeze(1) + self.point_in(feats)         # (N, P, ph)
            out = self.point_mlp(zz)                                              # (N, P, 3)
            mu = SIGMA_MAX * torch.tanh(out[..., 0])
            s = DELTA_SIGMA * (self.min_width + F.softplus(out[..., 1] + self.width_bias))
            s = s.clamp(max=10.0 * DELTA_SIGMA)
            sg = SIGMA.to(dev)
            G = torch.softmax(-0.5 * ((sg[None, None, :] - mu.unsqueeze(-1)) / s.unsqueeze(-1)) ** 2, dim=-1)
            w = torch.softmax(torch.log(expo + 1e-3) + self.g_scale * torch.tanh(out[..., 2]), dim=1)
            shape = (w.unsqueeze(-1) * G).sum(1)                                  # (N, 51), сумма = 1

            log_area = self.log_area_bias + self.area(h32).squeeze(-1)
            if self.use_geom_area:
                sasa = getattr(data, "sasa_ref", None)
                if sasa is None:
                    raise RuntimeError("sigma_surface: нужен data.sasa_ref (features: sasa / xtb_sasa)")
                log_area = log_area + self.log_area_scale * torch.log(sasa.float().clamp_min(1e-3))
            area = torch.exp(log_area.clamp(-20.0, 10.0))
            return area.unsqueeze(-1) * shape / DELTA_SIGMA
