"""Continuous-depth DimeNet++ encoder (Neural ODE).

The discrete backbone (backbones/dimenet_sigma.py) already computes, with N
SEPARATE interaction/output blocks:
    x_0 = edge_embedding
    P_0 = OutputBlock_0(x_0)
    for k in 1..N:
        x_k = InteractionBlock_k(x_{k-1})
        P_k = P_{k-1} + OutputBlock_k(x_k)
That accumulation is already an explicit-Euler discretisation of
    dP/dt = OutputBlock(x(t)),  dx/dt = InteractionBlock(x(t)) - x(t)
with step size 1 and a DIFFERENT block at every step. "Continuous" per the
architecture note means two real changes, not just adding another layer:
  1. ONE interaction block and ONE output block, reused at every t (weight
     tying across depth) - this is what actually changes the parameter count
     and inductive bias, not just how the loop is written.
  2. Integrate with an ODE solver (adaptive dopri5 via torchdiffeq if it is
     installed, else a manual fixed-step RK4 - see USE_TORCHDIFFEQ below) over
     t in [0, T] instead of a fixed sum.

State integrated jointly: (x, P) as a tuple, because dP/dt depends on x(t).

NOT verified end-to-end (no torch in this environment): py_compile only. The
`-x(t)` relaxation term is exactly what the architecture note specifies -
verify on your own smoke run that it does not just decay x to the trivial
fixed point x*=InteractionBlock(x*) too fast for T=1.0; if metrics come out
flat/undertrained, try increasing T first before adding depth via num_blocks.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer, InteractionPPBlock, OutputPPBlock, SphericalBasisLayer,
)
from torch_geometric.nn.resolver import activation_resolver

from ..data.constants import MAX_Z
from ..data.constants_3d import CUTOFF

logger = logging.getLogger(__name__)

try:
    from torchdiffeq import odeint as _torchdiffeq_odeint
    _HAVE_TORCHDIFFEQ = True
except ImportError:
    _HAVE_TORCHDIFFEQ = False
    logger.info("torchdiffeq not installed (pip install torchdiffeq --break-system-packages); "
               "falling back to a manual fixed-step RK4 integrator (see rk4_integrate below).")


def rk4_integrate(func, state0, t0: float, t1: float, n_steps: int):
    """Fixed-step RK4 over a tuple state, no external dependency.
    func(t, state) -> d(state)/dt, same tuple shape as state."""
    h = (t1 - t0) / n_steps
    state = state0
    t = t0
    for _ in range(n_steps):
        k1 = func(t, state)
        s2 = tuple(s + 0.5 * h * k for s, k in zip(state, k1))
        k2 = func(t + 0.5 * h, s2)
        s3 = tuple(s + 0.5 * h * k for s, k in zip(state, k2))
        k3 = func(t + 0.5 * h, s3)
        s4 = tuple(s + h * k for s, k in zip(state, k3))
        k4 = func(t + h, s4)
        state = tuple(s + (h / 6.0) * (a + 2 * b + 2 * c + d)
                     for s, a, b, c, d in zip(state, k1, k2, k3, k4))
        t += h
    return state


class _ODEFunc(nn.Module):
    """dx/dt = InteractionPPBlock(x) - x ; dP/dt = OutputPPBlock(x). Single
    shared pair of blocks, called at every solver step/substep (weight tying).
    rbf/sbf/idx_* are geometry, fixed for the whole trajectory (computed once
    per forward, not per ODE step) - passed in via closure, not the state."""

    def __init__(self, interaction: InteractionPPBlock, output: OutputPPBlock,
                rbf, sbf, idx_kj, idx_ji, i_idx, num_nodes):
        super().__init__()
        self.interaction, self.output = interaction, output
        self.rbf, self.sbf = rbf, sbf
        self.idx_kj, self.idx_ji, self.i_idx, self.num_nodes = idx_kj, idx_ji, i_idx, num_nodes

    def forward(self, t, state):
        x, P = state
        dx = self.interaction(x, self.rbf, self.sbf, self.idx_kj, self.idx_ji) - x
        dP = self.output(x, self.rbf, self.i_idx, num_nodes=self.num_nodes)
        return (dx, dP)


class ContinuousDimeNetPPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int = 118, out_emb_channels: int = 118,
                int_emb_size: int = 32, basis_emb_size: int = 8, num_spherical: int = 7,
                num_radial: int = 6, num_before_skip: int = 1, num_after_skip: int = 2,
                num_output_layers: int = 2, dropout: float = 0.05, cutoff: float = CUTOFF,
                envelope_exponent: int = 5, T: float = 1.0, n_steps: int = 6,
                use_torchdiffeq: bool = False, solver: str = "dopri5", **kwargs):
        super().__init__()
        self.T, self.n_steps = float(T), int(n_steps)
        self.use_torchdiffeq = bool(use_torchdiffeq) and _HAVE_TORCHDIFFEQ
        self.solver = solver
        act = activation_resolver("swish")

        self.z_embed = nn.Embedding(MAX_Z + 1, hidden_channels // 4)
        self.input_proj = nn.Linear(input_dim + hidden_channels // 4, hidden_channels)
        self.rbf_layer = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        self.edge_emb = nn.Sequential(nn.Linear(2 * (hidden_channels // 4) + num_radial, hidden_channels), nn.SiLU())
        # NOTE: nn.SiLU() here, not `act` from activation_resolver("swish") - in this
        # torch_geometric version activation_resolver returns a plain function, not an
        # nn.Module, and nn.Sequential requires every element to be a module. swish == SiLU,
        # so this is the same function, just wrapped correctly. `act` is still passed as-is
        # into InteractionPPBlock/OutputPPBlock below, which resolve it internally themselves
        # (that's how the original discrete backbone uses it too - only inside those blocks,
        # never directly inside a bare nn.Sequential).

        # ONE shared block pair, not num_blocks copies (weight tying across "depth")
        self.interaction = InteractionPPBlock(hidden_channels, int_emb_size, basis_emb_size,
                                              num_spherical, num_radial, num_before_skip,
                                              num_after_skip, act)
        self.output0 = OutputPPBlock(num_radial, hidden_channels, out_emb_channels, hidden_channels,
                                     num_output_layers, act)
        self.output = OutputPPBlock(num_radial, hidden_channels, out_emb_channels, hidden_channels,
                                    num_output_layers, act)
        self.dropout = nn.Dropout(dropout)
        self.rep_dim = out_emb_channels

    def encode(self, data: Data) -> torch.Tensor:
        pos, z = data.pos, data.z
        i, j = data.edge_index[1], data.edge_index[0]
        idx_i, idx_j, idx_k = data.tri_idx_i.long(), data.tri_idx_j.long(), data.tri_idx_k.long()
        idx_kj, idx_ji = data.tri_idx_kj.long(), data.tri_idx_ji.long()

        z_emb = self.z_embed(z.clamp(max=MAX_Z))
        h_atom = self.input_proj(torch.cat([data.x, z_emb], dim=-1))

        dist = data.edge_weight
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf_layer(dist)
        sbf = self.sbf_layer(dist, angle, idx_kj)
        x0 = self.edge_emb(torch.cat([h_atom[i], h_atom[j], rbf], dim=-1))
        P0 = self.output0(x0, rbf, i, num_nodes=pos.size(0))

        func = _ODEFunc(self.interaction, self.output, rbf, sbf, idx_kj, idx_ji, i, pos.size(0))
        if self.use_torchdiffeq:
            t = torch.tensor([0.0, self.T], device=pos.device, dtype=x0.dtype)
            xs, Ps = _torchdiffeq_odeint(func, (x0, P0), t, method=self.solver)
            xT, PT = xs[-1], Ps[-1]
        else:
            xT, PT = rk4_integrate(lambda t, s: func(t, s), (x0, P0), 0.0, self.T, self.n_steps)
        return self.dropout(PT)  # PT already includes P0 as the ODE's initial condition
