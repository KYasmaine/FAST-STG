# -*- coding: utf-8 -*-
"""
RF-SDE x-pred engine for conditional STG forecasting.

Key design:
- Only generate the future window, conditioned on history.
- Legacy raw mode bridges noise directly to x^{fut}.
- Location-scale mode bridges noise to r^{fut}=(x^{fut}-mu(c))/sigma(c),
  then maps the generated innovation back with mu(c)+sigma(c)*r^{fut}.
- Sampling uses Euler–Maruyama with pinning diffusion g(t)->0 as t->1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.diffstg.ugnet import UGnet


WeightSchedule = Literal["none", "jit"]  # jit: 1/(1-t)^2, equivalent to v-loss under x-parameterization
DiffusionSchedule = Literal["linear", "square"]  # g(t)=sigma*(1-t) or sigma*(1-t^2)
SamplingMode = Literal["sde", "ode"]
LossType = Literal["x", "v"]
SigmaMode = Literal["const", "cond_hist_std", "learned"]  # A1: history std; A2: learned sigma net
EndpointMode = Literal["zero", "lsnm"]  # zero: N(0, sigma), lsnm: N(mu(X), sigma(X))
MuMode = Literal["abs", "residual"]
TargetSpace = Literal["raw", "mean_centered", "location_scale"]
GenerationScope = Literal["future_only", "joint_history_future"]
EvalVariant = Literal["flow", "gaussian_ls"]
LSGraphMode = Literal["graph", "none"]


@dataclass
class RFSDEConfig:
    # sampling
    num_sampling_steps: int = 20
    sampling_mode: SamplingMode = "sde"
    sample_chunk: int = 10  # generate n_samples in chunks to reduce GPU memory

    # noise
    noise_scale: float = 1.0  # z0 ~ N(0, noise_scale^2 I)
    standard_source: bool = True  # same N(0,I) source in raw/residual ablations
    sigma: float = 0.05  # PEMS default; dataset-aware entry selects 0.08 for AIR
    g_schedule: DiffusionSchedule = "linear"
    sigma_mode: SigmaMode = "const"
    sigma_window: int = 0  # 0 => use full T_h; otherwise use last sigma_window steps from history
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    # A2 learned sigma head
    sigma_hidden: int = 128
    mu_hidden: int = 128
    mu_mode: MuMode = "residual"
    mu_scale: float = 0.1
    mu_clip: float = 0.0
    mu_loss_weight: float = 0.1
    sigma_target_window: int = 24  # trailing window length for pseudo-label from (history+future)
    sigma_loss_weight: float = 0.1  # auxiliary loss weight for sigma head
    sigma_log_loss: bool = True  # use log-space regression for stability
    # Step-3: joint eta_theta + sigma_theta learning
    joint_eta_sigma: bool = False
    joint_log_coeff: float = 1.0
    endpoint_mode: EndpointMode = "zero"

    # Conditional location-scale residual flow (second contribution).
    # raw: legacy future-space flow; mean_centered: flow on X_F-mu(c);
    # location_scale: flow on (X_F-mu(c))/sigma(c).
    target_space: TargetSpace = "location_scale"
    ls_hidden: int = 128
    ls_mu_scale: float = 0.1
    ls_sigma_min: float = 1e-3
    ls_sigma_max: float = 5.0
    ls_loss_weight: float = 1.0
    ls_detach_target: bool = True
    ls_warmup_epochs: int = 0  # deprecated alias for first-generation checkpoints
    ls_pretrain_epochs: int = 5
    ls_freeze_after_pretrain: bool = True
    # graph: bidirectional normalized adjacency propagation; none: identity
    # propagation, preserving the same projections/parameter count while
    # preventing information exchange between nodes.
    ls_graph_mode: LSGraphMode = "graph"

    # numerics
    delta: float = 1e-3  # numerical stabilizer for (1-t+delta)
    t_eps: float = 1e-6  # final clamp lower bound for denominator
    # UANS (uncertainty-aware noise schedule)
    uans_enable: bool = False
    uans_alpha: float = 0.0
    uans_min_pow: float = 0.5
    uans_max_pow: float = 2.0
    uans_floor: float = 0.0

    # loss
    weight_schedule: WeightSchedule = "none"
    loss_type: LossType = "x"  # x: x-pred MSE; v: v-pred MSE (JiT-style)
    transport_reduction: str = "sum"  # Eq. (13): squared L2 per trajectory

    # Controlled ablation for the first contribution.  future_only is the
    # proposed forecasting-aligned formulation; joint_history_future applies
    # the same RF-SDE and backbone to the complete history+future sequence.
    generation_scope: GenerationScope = "future_only"

    # gaussian_ls is the no-flow ablation: sample directly from the graph-aware
    # conditional Gaussian p(X_F|c)=N(mu(c), diag(sigma(c)^2)).
    eval_variant: EvalVariant = "flow"


class ConditionalLocationScaleHead(nn.Module):
    """Graph-aware history encoder with node/horizon-wise location-scale heads."""

    def __init__(
        self,
        stg_config,
        hidden: int,
        mu_mode: MuMode,
        mu_scale: float,
        mu_clip: float,
        sigma_min: float,
        sigma_max: float,
        graph_mode: LSGraphMode,
    ):
        super().__init__()
        self.T_h = int(stg_config.T_h)
        self.T_p = int(stg_config.T_p)
        self.F = int(stg_config.F)
        self.hidden = int(hidden)
        self.mu_mode = mu_mode
        self.mu_scale = float(mu_scale)
        self.mu_clip = float(mu_clip)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.graph_mode = str(graph_mode)

        if self.sigma_min <= 0:
            raise ValueError("ls_sigma_min must be positive")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("ls_sigma_max must be greater than ls_sigma_min")
        if self.graph_mode not in ("graph", "none"):
            raise ValueError(f"Unknown ls_graph_mode={self.graph_mode}")

        self.history_encoder = nn.Sequential(
            nn.Linear(self.T_h * self.F, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.self_proj = nn.Linear(self.hidden, self.hidden)
        self.forward_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.backward_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden)
        self.horizon_embedding = nn.Parameter(torch.empty(self.T_p, self.hidden))
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )
        self.mu_out = nn.Linear(self.hidden, self.F)
        self.scale_out = nn.Linear(self.hidden, self.F)

        adjacency = torch.tensor(np.asarray(stg_config.A, dtype=np.float32), dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Expected square adjacency, got {tuple(adjacency.shape)}")
        identity = torch.eye(adjacency.shape[0], dtype=adjacency.dtype)
        if self.graph_mode == "none":
            # Controlled graph ablation: retain all three projections but
            # replace neighbor aggregation with identity propagation. Each
            # node therefore depends only on its own history.
            adj_forward = identity
            adj_backward = identity
        else:
            adjacency = adjacency.clamp_min(0.0) + identity
            adj_forward = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            adj_backward = adjacency.transpose(0, 1)
            adj_backward = adj_backward / adj_backward.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("adj_forward", adj_forward)
        self.register_buffer("adj_backward", adj_backward)

        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)
        # Start from persistence and unit scale. This makes the coordinate
        # transform well behaved before the location-scale head is trained.
        nn.init.zeros_(self.mu_out.weight)
        nn.init.zeros_(self.mu_out.bias)
        nn.init.zeros_(self.scale_out.weight)
        unit_fraction = (1.0 - self.sigma_min) / (self.sigma_max - self.sigma_min)
        unit_fraction = min(max(unit_fraction, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.scale_out.bias, float(np.log(unit_fraction / (1.0 - unit_fraction))))

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """history (B,F,V,T_h) -> mu,sigma (B,F,V,T_p)."""
        B, Fdim, V, Th = history.shape
        if Fdim != self.F or Th != self.T_h:
            raise ValueError(
                f"History shape mismatch: expected F={self.F}, T_h={self.T_h}; "
                f"got {tuple(history.shape)}"
            )

        node_history = history.permute(0, 2, 3, 1).reshape(B, V, Th * Fdim)
        encoded = self.history_encoder(node_history)
        forward_context = torch.einsum("vw,bwh->bvh", self.adj_forward, encoded)
        backward_context = torch.einsum("vw,bwh->bvh", self.adj_backward, encoded)
        node_context = self.graph_norm(
            self.self_proj(encoded)
            + self.forward_proj(forward_context)
            + self.backward_proj(backward_context)
        )
        joint = node_context[:, :, None, :] + self.horizon_embedding[None, None, :, :]
        joint = self.decoder(joint)

        mu_delta = self.mu_out(joint).permute(0, 3, 1, 2)
        if self.mu_mode == "residual":
            persistence = history[..., -1:].expand(-1, -1, -1, self.T_p)
            mu = persistence + self.mu_scale * mu_delta
        else:
            mu = mu_delta
        if self.mu_clip > 0:
            mu = mu.clamp(min=-self.mu_clip, max=self.mu_clip)

        scale_logits = self.scale_out(joint).permute(0, 3, 1, 2)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(scale_logits)
        return mu.to(dtype=history.dtype), sigma.to(dtype=history.dtype)


class RFSDEXPred(nn.Module):
    """
    Rectified-flow style x-pred with SDE sampling (Euler–Maruyama).

    Input/Output conventions follow the existing codebase:
    - We take the same condition tuple c=(x_masked, pos_w, pos_d) where x_masked is (B,F,V,T),
      with T = T_h + T_p, containing (masked) history and zeros for future.
    - ``future_only`` samples only the future part and preserves the observed
      history in the output.
    - ``joint_history_future`` is a controlled ablation that transports the
      complete history+future sequence while keeping the same conditional
      backbone, RF objective and sampler.
    """

    def __init__(self, stg_config, rf_cfg: RFSDEConfig | None = None):
        super().__init__()
        self.stg_config = stg_config
        self.rf_cfg = rf_cfg or RFSDEConfig()
        if int(self.rf_cfg.num_sampling_steps) < 1:
            raise ValueError('num_sampling_steps must be positive')

        # reuse UGnet as backbone; it outputs (B,F,V,T_h+T_p)
        self.backbone = UGnet(stg_config)

        self.T_h = int(stg_config.T_h)
        self.T_p = int(stg_config.T_p)
        self.T = self.T_h + self.T_p
        self.F = int(stg_config.F)
        self.device = stg_config.device
        self.train_epoch = 0
        self.train_stage = "joint"
        self.last_loss_components = {}

        # A2: learned sigma network g_psi(x^{his}) -> sigma for each (node, future step)
        if self.rf_cfg.sigma_mode == "learned":
            in_dim = self.T_h * self.F
            hid = int(self.rf_cfg.sigma_hidden)
            self.sigma_net = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, self.T_p),
            )
        else:
            self.sigma_net = None

        # LSNM endpoint mean head: f_phi(x^{his}) -> mu for each (node, future step)
        if self.rf_cfg.endpoint_mode == "lsnm":
            in_dim = self.T_h * self.F
            hid = int(self.rf_cfg.mu_hidden)
            self.mu_net = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, self.T_p),
            )
        else:
            self.mu_net = None

        target_space = str(getattr(self.rf_cfg, "target_space", "raw"))
        if target_space not in ("raw", "mean_centered", "location_scale"):
            raise ValueError(f"Unknown target_space={target_space}")
        if target_space != "raw":
            if self.rf_cfg.endpoint_mode != "zero":
                raise ValueError("Residual target spaces require endpoint_mode='zero'")
            if self.rf_cfg.sigma_mode != "const":
                raise ValueError("Residual target spaces require sigma_mode='const'; sigma(c) is learned by the LS head")
            if bool(self.rf_cfg.uans_enable):
                raise ValueError("Residual target spaces do not use UANS")
            self.location_scale_head = ConditionalLocationScaleHead(
                stg_config=stg_config,
                hidden=int(self.rf_cfg.ls_hidden),
                mu_mode=self.rf_cfg.mu_mode,
                mu_scale=float(self.rf_cfg.ls_mu_scale),
                mu_clip=float(self.rf_cfg.mu_clip),
                sigma_min=float(self.rf_cfg.ls_sigma_min),
                sigma_max=float(self.rf_cfg.ls_sigma_max),
                graph_mode=str(getattr(self.rf_cfg, "ls_graph_mode", "graph")),
            )
        else:
            self.location_scale_head = None

        generation_scope = str(
            getattr(self.rf_cfg, "generation_scope", "future_only")
        )
        if generation_scope not in ("future_only", "joint_history_future"):
            raise ValueError(f"Unknown generation_scope={generation_scope}")
        if generation_scope == "joint_history_future":
            if target_space != "raw":
                raise ValueError(
                    "joint_history_future is a contribution-1 ablation and "
                    "requires target_space='raw'"
                )
            if self.rf_cfg.sigma_mode != "const":
                raise ValueError(
                    "joint_history_future currently requires sigma_mode='const'"
                )
            if self.rf_cfg.endpoint_mode != "zero":
                raise ValueError(
                    "joint_history_future currently requires endpoint_mode='zero'"
                )

        eval_variant = str(getattr(self.rf_cfg, "eval_variant", "flow"))
        if eval_variant not in ("flow", "gaussian_ls"):
            raise ValueError(f"Unknown eval_variant={eval_variant}")
        if eval_variant == "gaussian_ls" and target_space != "location_scale":
            raise ValueError(
                "gaussian_ls evaluation requires target_space='location_scale'"
            )
        ls_graph_mode = str(getattr(self.rf_cfg, "ls_graph_mode", "graph"))
        if ls_graph_mode not in ("graph", "none"):
            raise ValueError(f"Unknown ls_graph_mode={ls_graph_mode}")

    # --- compatibility with existing training script interface ---
    def model_file_name(self):
        # keep a stable suffix so train.py can build checkpoint paths
        base = f"RFSDE+Th-{self.T_h}+Tp-{self.T_p}+steps-{self.rf_cfg.num_sampling_steps}+mode-{self.rf_cfg.sampling_mode}"
        target_space = str(getattr(self.rf_cfg, "target_space", "raw"))
        if target_space != "raw":
            base = f"{base}+target-{target_space}"
        if self._generation_scope() != "future_only":
            base = f"{base}+scope-{self._generation_scope()}"
        if self._eval_variant() != "flow":
            base = f"{base}+eval-{self._eval_variant()}"
        if target_space != "raw" and self._ls_graph_mode() != "graph":
            base = f"{base}+lsgraph-{self._ls_graph_mode()}"
        return f"{base}.rfstg"

    def _target_space(self) -> str:
        """Read safely from old pickled checkpoints that predate target_space."""
        return str(getattr(self.rf_cfg, "target_space", "raw"))

    def _generation_scope(self) -> str:
        """Read safely from checkpoints created before the ablation switch."""
        return str(getattr(self.rf_cfg, "generation_scope", "future_only"))

    def _eval_variant(self) -> str:
        """Return the inference distribution used by this checkpoint/run."""
        return str(getattr(self.rf_cfg, "eval_variant", "flow"))

    def _ls_graph_mode(self) -> str:
        """Read safely from checkpoints created before the graph ablation."""
        return str(getattr(self.rf_cfg, "ls_graph_mode", "graph"))

    def set_train_epoch(self, epoch: int) -> str:
        """Configure strict LS-pretrain or residual-flow parameter freezing."""
        self.train_epoch = int(epoch)
        target_space = self._target_space()
        if target_space == "raw":
            self.train_stage = "joint"
            return self.train_stage

        pretrain_epochs = int(
            getattr(
                self.rf_cfg,
                "ls_pretrain_epochs",
                getattr(self.rf_cfg, "ls_warmup_epochs", 0),
            )
        )
        freeze_after = bool(getattr(self.rf_cfg, "ls_freeze_after_pretrain", True))
        if self.train_epoch < pretrain_epochs:
            self.train_stage = "location_scale_pretrain"
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in self.location_scale_head.parameters():
                parameter.requires_grad_(True)
        elif freeze_after:
            self.train_stage = "residual_flow"
            for parameter in self.location_scale_head.parameters():
                parameter.requires_grad_(False)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)
        else:
            self.train_stage = "joint"
            for parameter in self.location_scale_head.parameters():
                parameter.requires_grad_(True)
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)
        return self.train_stage

    def get_train_stage(self) -> str:
        return str(getattr(self, "train_stage", "joint"))

    def _transport_loss(self, squared_error: torch.Tensor) -> torch.Tensor:
        reduction = getattr(self.rf_cfg, "transport_reduction", "mean")
        if reduction == "sum":
            return squared_error.flatten(1).sum(dim=1).mean()
        if reduction == "mean":
            return squared_error.mean()
        raise ValueError(f"Unknown transport_reduction={reduction}")

    def _compute_location_scale(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.location_scale_head is None:
            raise RuntimeError("location_scale_head is only available for residual target spaces")
        return self.location_scale_head(history)

    def _encode_future_target(
        self,
        future: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        detach: bool,
    ) -> torch.Tensor:
        if detach:
            mu = mu.detach()
            sigma = sigma.detach()
        target_space = self._target_space()
        if target_space == "mean_centered":
            return future - mu
        if target_space == "location_scale":
            return (future - mu) / sigma.clamp_min(float(self.rf_cfg.ls_sigma_min))
        return future

    def _decode_future_target(
        self, target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        target_space = self._target_space()
        if target_space == "mean_centered":
            return mu + target
        if target_space == "location_scale":
            return mu + sigma * target
        return target

    @staticmethod
    def _gaussian_location_scale_nll(
        future: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-8)
        valid = torch.isfinite(future)
        safe_future = torch.where(valid, future, mu.detach())
        standardized = (safe_future - mu) / sigma
        scores = 0.5 * standardized.square() + torch.log(sigma)
        if not valid.any():
            raise ValueError('No observed future targets in minibatch')
        return scores[valid].mean()

    def set_ddim_sample_steps(self, _sample_steps: int):
        # no-op (kept for train.py compatibility)
        return

    def set_sample_strategy(self, _sample_strategy: str):
        # no-op (kept for train.py compatibility)
        return

    def _split_history_future(self, x_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_full: (B,F,V,T) -> (history:(B,F,V,T_h), future:(B,F,V,T_p))"""
        h = x_full[..., : self.T_h]
        f = x_full[..., self.T_h : self.T_h + self.T_p]
        return h, f

    def _build_full_inputs(self, history: torch.Tensor, future_like: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_full, x_masked_full) both (B,F,V,T)."""
        x_full = torch.cat([history, future_like], dim=-1)
        x_masked_full = torch.cat([history, torch.zeros_like(future_like)], dim=-1)
        return x_full, x_masked_full

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,1,1,1) in [0,1]
        returns w(t) broadcastable to loss tensor (B,F,V,T_p).
        """
        if self.rf_cfg.weight_schedule == "none":
            return torch.ones_like(t)
        if self.rf_cfg.weight_schedule == "jit":
            # Equivalent to v-loss: ||v - v_pred||^2 = ||x-x_pred||^2 / (1-t)^2
            denom = (1.0 - t + self.rf_cfg.delta).clamp_min(self.rf_cfg.t_eps)
            return 1.0 / (denom * denom)
        raise ValueError(f"Unknown weight_schedule={self.rf_cfg.weight_schedule}")

    def _compute_sigma_from_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: (B,F,V,T_h)
        returns sigma_cond: (B,1,V,1) (broadcastable to future (B,F,V,T_p))
        A1: sigma_cond = base_sigma * std(history over time [+feature]), clamped.
        """
        # choose window
        if self.rf_cfg.sigma_window and int(self.rf_cfg.sigma_window) > 0:
            w = int(self.rf_cfg.sigma_window)
            hist = history[..., -w:]
        else:
            hist = history

        # std over time and feature dims -> (B,1,V,1)
        # hist: (B,F,V,Th) -> (B,V)
        std = hist.float().std(dim=(-1, 1), unbiased=False)  # (B,V)
        sigma = float(self.rf_cfg.sigma) * std  # scale by base sigma
        sigma = sigma.clamp(min=float(self.rf_cfg.sigma_min), max=float(self.rf_cfg.sigma_max))
        return sigma[:, None, :, None].to(dtype=history.dtype, device=history.device)

    def _compute_sigma_learned(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: (B,F,V,T_h) -> sigma_pred: (B,1,V,T_p)
        """
        assert self.sigma_net is not None
        B, Fdim, V, Th = history.shape
        x = history.permute(0, 2, 3, 1).reshape(B, V, Th * Fdim)  # (B,V,Th*F)
        raw = self.sigma_net(x)  # (B,V,T_p)
        # A2 stabilized: learn a *relative* uncertainty factor around 1, keep global scale at base sigma.
        # factor in (0.5, 1.5) by default => avoids blowing up noise magnitude.
        factor = 1.0 + 0.5 * torch.tanh(raw)
        sigma = float(self.rf_cfg.sigma) * factor
        sigma = sigma.clamp(min=float(self.rf_cfg.sigma_min), max=float(self.rf_cfg.sigma_max))
        return sigma[:, None, :, :].to(dtype=history.dtype, device=history.device)

    def _compute_mu_from_history(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: (B,F,V,T_h) -> mu_pred: (B,1,V,T_p)
        """
        B, Fdim, V, Th = history.shape
        if self.rf_cfg.endpoint_mode != "lsnm":
            return torch.zeros((B, 1, V, self.T_p), dtype=history.dtype, device=history.device)
        assert self.mu_net is not None

        # baseline mean: naive persistence from the last history step, broadcast to future horizon
        mu_base = history[..., -1:].mean(dim=1, keepdim=True).expand(-1, 1, -1, self.T_p)
        x = history.permute(0, 2, 3, 1).reshape(B, V, Th * Fdim)  # (B,V,Th*F)
        mu_raw = self.mu_net(x)[:, None, :, :]  # (B,1,V,T_p)

        if self.rf_cfg.mu_mode == "residual":
            mu = mu_base + float(self.rf_cfg.mu_scale) * mu_raw
        else:
            mu = mu_raw

        clip = float(self.rf_cfg.mu_clip)
        if clip > 0:
            mu = mu.clamp(min=-clip, max=clip)
        return mu.to(dtype=history.dtype, device=history.device)

    @staticmethod
    def _trailing_var(x: torch.Tensor, window: int) -> torch.Tensor:
        """
        x: (B,T,V) -> var: (B,T,V) where var[t] computed on trailing window [t-window, t-1]
        (NsDiff-style wv_sigma_trailing). Uses replicate padding at start.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        B, T, V = x.shape
        if window > T:
            window = T
        x_pad = F.pad(x, (0, 0, window, 0), mode="replicate")  # (B,T+window,V)
        # unfold adds the window dimension at the end: (B, T, V, window)
        windows = x_pad.unfold(1, window, 1)
        # variance across the trailing-window dimension (last dim)
        var = windows.var(dim=3, unbiased=False)  # (B,T,V)
        return var

    def _sigma_target_from_truth(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo-label sigma target from ground-truth (history+future) using trailing window variance.
        x_full: (B,F,V,T) -> sigma_target: (B,1,V,T_p)
        """
        B, Fdim, V, T = x_full.shape
        # collapse feature dim (F) since traffic is typically F=1; keep mean for generality
        x = x_full.mean(dim=1)  # (B,V,T)
        x = x.transpose(1, 2)  # (B,T,V)
        var = self._trailing_var(x, int(self.rf_cfg.sigma_target_window))  # (B,T,V)
        var_fut = var[:, -self.T_p :, :]  # (B,T_p,V)
        sigma = torch.sqrt(var_fut + 1e-8)  # std
        sigma = sigma.transpose(1, 2)  # (B,V,T_p)
        sigma = sigma.clamp(min=float(self.rf_cfg.sigma_min), max=float(self.rf_cfg.sigma_max))
        return sigma[:, None, :, :].to(dtype=x_full.dtype, device=x_full.device)

    def _g(self, t_scalar: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        g(t) schedule with pinning: g(t)->0 as t->1.
        t_scalar: (B,1,1,1) or (B,...) in [0,1]
        sigma: broadcastable to t_scalar and z (e.g., (B,1,V,1) or scalar-like)
        """
        if self.rf_cfg.sampling_mode == "ode":
            return torch.zeros_like(t_scalar)

        if self.rf_cfg.g_schedule == "linear":
            base = (1.0 - t_scalar).clamp_min(self.rf_cfg.t_eps)
        elif self.rf_cfg.g_schedule == "square":
            base = (1.0 - t_scalar * t_scalar).clamp_min(self.rf_cfg.t_eps)
        else:
            raise ValueError(f"Unknown g_schedule={self.rf_cfg.g_schedule}")

        # UANS: high uncertainty -> slower decay (larger g near terminal region).
        if bool(self.rf_cfg.uans_enable) and float(self.rf_cfg.uans_alpha) > 0:
            smin = float(self.rf_cfg.sigma_min)
            smax = float(self.rf_cfg.sigma_max)
            u = ((sigma - smin) / (smax - smin + 1e-8)).clamp(0.0, 1.0)
            p = (1.0 - float(self.rf_cfg.uans_alpha) * u).clamp(
                min=float(self.rf_cfg.uans_min_pow), max=float(self.rf_cfg.uans_max_pow)
            )
            decay = base.pow(p)
            floor = float(self.rf_cfg.uans_floor)
            if floor > 0:
                decay = floor + (1.0 - floor) * decay
            return sigma * decay

        return sigma * base

    def predict_x_fut(self, z_fut: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        """
        Predict clean future x^{fut} given current noisy future z_fut and condition (history).
        z_fut: (B,F,V,T_p)
        t: (B,1,1,1) in [0,1]
        cond: (x_masked_full, pos_w, pos_d) where x_masked_full contains history.
        """
        x_masked_full, pos_w, pos_d = cond
        history, _ = self._split_history_future(x_masked_full)

        x_full, x_masked_full2 = self._build_full_inputs(history, z_fut)
        # reuse the same condition form UGnet expects: (x_masked, pos_w, pos_d)
        x_pred_full = self.backbone(x_full, t.flatten(), (x_masked_full2, pos_w, pos_d))
        _, x_pred_fut = self._split_history_future(x_pred_full)
        return x_pred_fut

    def loss(self, x_full: torch.Tensor, cond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        x_full: (B,F,V,T) ground-truth full sequence (history+future)
        cond: (x_masked_full, pos_w, pos_d) with masked history and zero future.
        """
        _history_true, x_fut_true = self._split_history_future(x_full)
        observed_full = torch.isfinite(x_full)
        observed_future = torch.isfinite(x_fut_true)
        x_full = torch.where(observed_full, x_full, torch.zeros_like(x_full))
        safe_future = torch.where(observed_future, x_fut_true, torch.zeros_like(x_fut_true))

        B = x_full.shape[0]
        t = torch.rand((B, 1, 1, 1), device=x_full.device, dtype=x_full.dtype)

        # A1: non-stationary sigma from history (condition) to scale endpoint noise
        x_masked_full, _pos_w, _pos_d = cond
        history_cond, _ = self._split_history_future(x_masked_full)

        # Contribution-1 controlled ablation: use the same RF-SDE formulation
        # and conditional UGnet, but transport the complete observed-history +
        # future target instead of restricting transport to the future space.
        if self._generation_scope() == "joint_history_future":
            sigma = torch.tensor(
                float(self.rf_cfg.sigma),
                device=x_full.device,
                dtype=x_full.dtype,
            ).view(1, 1, 1, 1)
            endpoint = torch.randn_like(x_full) * self.rf_cfg.noise_scale
            if not getattr(self.rf_cfg, "standard_source", False):
                endpoint = endpoint * sigma
            z_t = t * x_full + (1.0 - t) * endpoint
            x_pred_full = self.backbone(
                z_t, t.flatten(), (x_masked_full, _pos_w, _pos_d)
            )
            denom = (1.0 - t + self.rf_cfg.delta).clamp_min(
                self.rf_cfg.t_eps
            )
            if self.rf_cfg.loss_type == "x":
                loss_main = self._transport_loss(
                    self._weight(t) * (x_pred_full - x_full).square() * observed_full
                )
            elif self.rf_cfg.loss_type == "v":
                velocity_true = (x_full - z_t) / denom
                velocity_pred = (x_pred_full - z_t) / denom
                loss_main = self._transport_loss((velocity_true - velocity_pred).square() * observed_full)
            else:
                raise ValueError(f"Unknown loss_type={self.rf_cfg.loss_type}")
            self.last_loss_components = {
                "flow": loss_main.detach(),
                "total": loss_main.detach(),
            }
            return loss_main

        # New method: define the bridge entirely in a standardized innovation
        # space. The LS head is optimized by a proper scoring rule, while the
        # stop-gradient transform prevents it from inflating sigma merely to
        # simplify the flow target.
        target_space = self._target_space()
        if target_space != "raw":
            mu_cond, scale_cond = self._compute_location_scale(history_cond)
            # Keep estimator training identical in the mean-centered ablation;
            # only omit scale normalization in the transport coordinates.
            loss_ls = self._gaussian_location_scale_nll(x_fut_true, mu_cond, scale_cond)

            train_stage = self.get_train_stage()
            if train_stage == "location_scale_pretrain":
                self.last_loss_components = {
                    "flow": torch.zeros_like(loss_ls.detach()),
                    "location_scale": loss_ls.detach(),
                    "total": loss_ls.detach(),
                    "sigma_mean": scale_cond.detach().mean(),
                }
                return loss_ls

            target_fut = self._encode_future_target(
                safe_future,
                mu_cond,
                scale_cond,
                detach=bool(self.rf_cfg.ls_detach_target),
            )
            target_fut = torch.where(observed_future, target_fut, torch.zeros_like(target_fut))
            endpoint = torch.randn_like(target_fut) * self.rf_cfg.noise_scale
            z_t = t * target_fut + (1.0 - t) * endpoint
            target_pred = self.predict_x_fut(z_t, t, cond)
            denom = (1.0 - t + self.rf_cfg.delta).clamp_min(self.rf_cfg.t_eps)

            if self.rf_cfg.loss_type == "x":
                loss_main = self._transport_loss(self._weight(t) * (target_pred - target_fut).square() * observed_future)
            elif self.rf_cfg.loss_type == "v":
                velocity_true = (target_fut - z_t) / denom
                velocity_pred = (target_pred - z_t) / denom
                loss_main = self._transport_loss((velocity_true - velocity_pred).square() * observed_future)
            else:
                raise ValueError(f"Unknown loss_type={self.rf_cfg.loss_type}")

            if train_stage == "residual_flow":
                loss_total = loss_main
            else:
                # Backward-compatible joint mode for ablations only.
                loss_total = loss_main + float(self.rf_cfg.ls_loss_weight) * loss_ls
            self.last_loss_components = {
                "flow": loss_main.detach(),
                "location_scale": loss_ls.detach(),
                "total": loss_total.detach(),
                "sigma_mean": scale_cond.detach().mean(),
            }
            return loss_total

        if self.rf_cfg.sigma_mode == "cond_hist_std":
            sigma_cond = self._compute_sigma_from_history(history_cond)  # (B,1,V,1)
        elif self.rf_cfg.sigma_mode == "learned":
            sigma_cond = self._compute_sigma_learned(history_cond)  # (B,1,V,T_p)
        else:
            sigma_cond = torch.tensor(float(self.rf_cfg.sigma), device=x_full.device, dtype=x_full.dtype).view(1, 1, 1, 1)

        mu_cond = self._compute_mu_from_history(history_cond)  # (B,1,V,T_p) or zeros
        x_fut_true = safe_future
        endpoint = torch.randn_like(x_fut_true) * self.rf_cfg.noise_scale
        if not getattr(self.rf_cfg, "standard_source", False):
            endpoint = mu_cond + endpoint * sigma_cond
        z_t = t * x_fut_true + (1.0 - t) * endpoint

        x_fut_pred = self.predict_x_fut(z_t, t, cond)

        denom = (1.0 - t + self.rf_cfg.delta).clamp_min(self.rf_cfg.t_eps)

        if self.rf_cfg.loss_type == "x":
            w = self._weight(t)
            err = x_fut_pred - x_fut_true
            # Step-3 (optional): heteroscedastic main loss to couple eta_theta and sigma_theta.
            if self.rf_cfg.sigma_mode == "learned" and bool(self.rf_cfg.joint_eta_sigma):
                sigma_for_loss = sigma_cond.clamp(min=float(self.rf_cfg.sigma_min), max=float(self.rf_cfg.sigma_max))
                inv_var = 1.0 / (sigma_for_loss * sigma_for_loss + 1e-8)
                nll = err * err * inv_var + float(self.rf_cfg.joint_log_coeff) * torch.log(sigma_for_loss * sigma_for_loss + 1e-8)
                loss_main = (w * nll).mean()
            else:
                loss = w * (err ** 2) * observed_future
                loss_main = self._transport_loss(loss)
            # A2: auxiliary sigma regression (NsDiff-style) to prevent trivial sigma collapse
            loss_mu = torch.tensor(0.0, device=x_full.device, dtype=x_full.dtype)
            if self.rf_cfg.endpoint_mode == "lsnm" and float(self.rf_cfg.mu_loss_weight) > 0:
                # Directly supervise f(X) to align with conditional mean target.
                loss_mu = (mu_cond - x_fut_true).pow(2).mean()

            if self.rf_cfg.sigma_mode == "learned" and self.rf_cfg.sigma_loss_weight > 0:
                sigma_pred = sigma_cond  # (B,1,V,Tp)
                sigma_tgt = self._sigma_target_from_truth(x_full).detach()
                # Robustness: ensure sigma_tgt matches sigma_pred layout (B,1,V,Tp)
                if sigma_tgt.shape != sigma_pred.shape:
                    # common mismatch: (B,1,Tp,V) vs (B,1,V,Tp)
                    if (
                        sigma_tgt.ndim == 4
                        and sigma_pred.ndim == 4
                        and sigma_tgt.shape[0] == sigma_pred.shape[0]
                        and sigma_tgt.shape[1] == sigma_pred.shape[1]
                        and sigma_tgt.shape[2] == sigma_pred.shape[3]
                        and sigma_tgt.shape[3] == sigma_pred.shape[2]
                    ):
                        sigma_tgt = sigma_tgt.transpose(2, 3)
                    else:
                        raise RuntimeError(
                            f"sigma shape mismatch: pred={tuple(sigma_pred.shape)} tgt={tuple(sigma_tgt.shape)}"
                        )
                # Learn *relative* heteroscedasticity: normalize target by its own mean to avoid absolute-scale mismatch.
                base = float(self.rf_cfg.sigma)
                factor_pred = (sigma_pred / (base + 1e-8)).clamp_min(1e-6)
                # normalize target to mean-1 per sample
                tgt_mean = sigma_tgt.mean(dim=(2, 3), keepdim=True).clamp_min(1e-8)
                factor_tgt = (sigma_tgt / tgt_mean).clamp_min(1e-6)
                if self.rf_cfg.sigma_log_loss:
                    loss_sigma = (torch.log(factor_pred) - torch.log(factor_tgt)) ** 2
                else:
                    loss_sigma = (factor_pred - factor_tgt) ** 2
                loss_sigma = loss_sigma.mean()
                return (
                    loss_main
                    + float(self.rf_cfg.sigma_loss_weight) * loss_sigma
                    + float(self.rf_cfg.mu_loss_weight) * loss_mu
                )
            return loss_main + float(self.rf_cfg.mu_loss_weight) * loss_mu

        if self.rf_cfg.loss_type == "v":
            # JiT-style: v = (x - z)/(1-t), v_pred = (x_pred - z)/(1-t)
            v_true = (x_fut_true - z_t) / denom
            v_pred = (x_fut_pred - z_t) / denom
            return ((v_true - v_pred) ** 2).mean()

        raise ValueError(f"Unknown loss_type={self.rf_cfg.loss_type}")

    @torch.no_grad()
    def sample_future(self, cond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
        """
        Return samples for full sequence: (B,n_samples,F,V,T) where history is preserved from condition.
        """
        if n_samples < 1:
            raise ValueError('n_samples must be positive')
        if self._eval_variant() == "gaussian_ls":
            return self._sample_gaussian_location_scale(cond, n_samples=n_samples)
        if self._generation_scope() == "joint_history_future":
            return self._sample_joint_history_future(cond, n_samples=n_samples)

        x_masked_full, pos_w, pos_d = cond
        history, _ = self._split_history_future(x_masked_full)

        B, F, V, _T = x_masked_full.shape
        assert _T == self.T, f"Expected T={self.T}, got {_T}"

        K = int(self.rf_cfg.num_sampling_steps)
        dt = 1.0 / K

        chunk = int(self.rf_cfg.sample_chunk) if self.rf_cfg.sample_chunk else n_samples
        chunk = max(1, min(chunk, n_samples))

        # Algorithm 2 computes the deterministic coordinates once per history.
        coordinates = self._compute_location_scale(history) if self._target_space() != "raw" else None

        outs = []
        for start in range(0, n_samples, chunk):
            cur = min(chunk, n_samples - start)

            history_rep = history.unsqueeze(1).repeat(1, cur, 1, 1, 1).reshape(-1, F, V, self.T_h)
            pos_w_rep = pos_w.unsqueeze(1).repeat(1, cur, *([1] * (pos_w.ndim - 1))).reshape(-1, *pos_w.shape[1:])
            pos_d_rep = pos_d.unsqueeze(1).repeat(1, cur, *([1] * (pos_d.ndim - 1))).reshape(-1, *pos_d.shape[1:])

            # Initial future state. Residual modes always start from a standard
            # innovation source; the learned conditional scale is used only in
            # the target transform and final inverse transform.
            z = torch.randn((B * cur, F, V, self.T_p), device=self.device) * self.rf_cfg.noise_scale
            target_space = self._target_space()
            if target_space != "raw":
                mu_cond, scale_cond = (
                    value.unsqueeze(1).expand(-1, cur, -1, -1, -1).reshape(B * cur, F, V, self.T_p)
                    for value in coordinates
                )
                diffusion_sigma = torch.tensor(
                    float(self.rf_cfg.sigma), device=self.device, dtype=z.dtype
                ).view(1, 1, 1, 1)
            else:
                # Legacy raw-space endpoint behavior.
                if self.rf_cfg.sigma_mode == "cond_hist_std":
                    sigma_cond = self._compute_sigma_from_history(history_rep)  # (B*cur,1,V,1)
                elif self.rf_cfg.sigma_mode == "learned":
                    sigma_cond = self._compute_sigma_learned(history_rep)  # (B*cur,1,V,T_p)
                else:
                    sigma_cond = torch.tensor(
                        float(self.rf_cfg.sigma), device=self.device, dtype=z.dtype
                    ).view(1, 1, 1, 1)
                mu_cond = self._compute_mu_from_history(history_rep)  # (B*cur,1,V,T_p) or zeros
                scale_cond = sigma_cond
                diffusion_sigma = sigma_cond
                if not getattr(self.rf_cfg, "standard_source", False):
                    z = mu_cond + z * sigma_cond

            for k in range(K):
                t0 = torch.full((B * cur, 1, 1, 1), k * dt, device=self.device, dtype=z.dtype)
                denom = (1.0 - t0 + self.rf_cfg.delta).clamp_min(self.rf_cfg.t_eps)

                x_full, x_masked_full2 = self._build_full_inputs(history_rep, z)
                x_pred_full = self.backbone(x_full, t0.flatten(), (x_masked_full2, pos_w_rep, pos_d_rep))
                _, x_pred_fut = self._split_history_future(x_pred_full)

                v = (x_pred_fut - z) / denom
                z = z + dt * v

                g = self._g(t0, sigma=diffusion_sigma)
                if self.rf_cfg.sampling_mode == "sde":
                    z = z + g * (dt ** 0.5) * torch.randn_like(z)

            future_out = self._decode_future_target(z, mu_cond, scale_cond)
            x_out = torch.cat([history_rep, future_out], dim=-1)  # (B*cur,F,V,T)
            x_out = x_out.reshape(B, cur, F, V, self.T).detach().cpu()
            outs.append(x_out)

            torch.cuda.empty_cache()

        # Keep samples on CPU to reduce GPU memory pressure during evaluation; caller can move it if needed.
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def _sample_gaussian_location_scale(
        self,
        cond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample the graph-aware conditional Gaussian without invoking Flow."""
        if self.location_scale_head is None or self._target_space() != "location_scale":
            raise RuntimeError(
                "Gaussian LS sampling requires a location_scale checkpoint"
            )
        x_masked_full, _pos_w, _pos_d = cond
        history, _ = self._split_history_future(x_masked_full)
        B, F, V, T = x_masked_full.shape
        if T != self.T:
            raise ValueError(f"Expected T={self.T}, got {T}")

        chunk = int(self.rf_cfg.sample_chunk) if self.rf_cfg.sample_chunk else n_samples
        chunk = max(1, min(chunk, n_samples))
        outs = []
        for start in range(0, n_samples, chunk):
            cur = min(chunk, n_samples - start)
            history_rep = (
                history.unsqueeze(1)
                .repeat(1, cur, 1, 1, 1)
                .reshape(-1, F, V, self.T_h)
            )
            mu, sigma = self._compute_location_scale(history_rep)
            future = mu + sigma * torch.randn_like(mu)
            full = torch.cat([history_rep, future], dim=-1)
            outs.append(full.reshape(B, cur, F, V, self.T).detach().cpu())
        return torch.cat(outs, dim=1)

    @torch.no_grad()
    def _sample_joint_history_future(
        self,
        cond: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample the full sequence for the contribution-1 joint ablation."""
        x_masked_full, pos_w, pos_d = cond
        B, F, V, T = x_masked_full.shape
        if T != self.T:
            raise ValueError(f"Expected T={self.T}, got {T}")

        K = int(self.rf_cfg.num_sampling_steps)
        dt = 1.0 / K
        chunk = int(self.rf_cfg.sample_chunk) if self.rf_cfg.sample_chunk else n_samples
        chunk = max(1, min(chunk, n_samples))
        sigma = torch.tensor(
            float(self.rf_cfg.sigma),
            device=self.device,
            dtype=x_masked_full.dtype,
        ).view(1, 1, 1, 1)

        outs = []
        for start in range(0, n_samples, chunk):
            cur = min(chunk, n_samples - start)
            x_masked_rep = (
                x_masked_full.unsqueeze(1)
                .repeat(1, cur, 1, 1, 1)
                .reshape(-1, F, V, T)
            )
            pos_w_rep = (
                pos_w.unsqueeze(1)
                .repeat(1, cur, *([1] * (pos_w.ndim - 1)))
                .reshape(-1, *pos_w.shape[1:])
            )
            pos_d_rep = (
                pos_d.unsqueeze(1)
                .repeat(1, cur, *([1] * (pos_d.ndim - 1)))
                .reshape(-1, *pos_d.shape[1:])
            )

            z = (
                torch.randn((B * cur, F, V, T), device=self.device)
                * self.rf_cfg.noise_scale
            )
            if not getattr(self.rf_cfg, "standard_source", False):
                z = z * sigma
            joint_cond = (x_masked_rep, pos_w_rep, pos_d_rep)
            for k in range(K):
                t0 = torch.full(
                    (B * cur, 1, 1, 1),
                    k * dt,
                    device=self.device,
                    dtype=z.dtype,
                )
                denom = (1.0 - t0 + self.rf_cfg.delta).clamp_min(
                    self.rf_cfg.t_eps
                )
                x_pred_full = self.backbone(
                    z, t0.flatten(), joint_cond
                )
                z = z + dt * (x_pred_full - z) / denom
                if self.rf_cfg.sampling_mode == "sde":
                    z = z + self._g(t0, sigma=sigma) * (dt ** 0.5) * torch.randn_like(z)

            outs.append(
                z.reshape(B, cur, F, V, T).detach().cpu()
            )
            torch.cuda.empty_cache()

        return torch.cat(outs, dim=1)

    def forward(self, input, n_samples: int = 1) -> torch.Tensor:
        # Keep the same interface as DiffSTG: model((x_masked,pos_w,pos_d), n_samples) -> (B,n_samples,F,V,T)
        return self.sample_future(input, n_samples)


# Backward-compatible alias for train.py (it already references RFSDE_XPred)
RFSDE_XPred = RFSDEXPred
