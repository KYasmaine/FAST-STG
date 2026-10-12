# -*- coding: utf-8 -*-
import os, sys
import json
import hashlib
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer

from utils.eval import (
    ConditionalResidualDiagnosticAccumulator,
    CoverageSpreadCalibrator,
    Metric,
    StreamingMetricAccumulator,
    apply_residual_spread_calibration,
    resolve_eval_n_samples,
)
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file
from algorithm.rf_sde.model import RFSDE_XPred
# Default manuscript experiment settings and checkpoint compatibility checks.
PAPER_VARIANTS = (
    "main", "ablation_full", "joint_window_raw", "future_only_raw",
    "mean_centered", "no_graph", "no_freezing",
)
PROTOCOL_VERSION = "www2027-tex-806fd0f7-v1"


def apply_www2027_protocol(params):
    variant = params.get("paper_variant", "main")
    experiment = params.get("paper_experiment", "standard")
    if experiment not in ("standard", "sensitivity", "diagnostic"):
        raise ValueError(f"Unknown paper experiment: {experiment}")
    if variant not in PAPER_VARIANTS:
        raise ValueError(f"Unknown paper variant: {variant}")
    params.update(
        engine="rf_sde_xpred", rf_target_space="location_scale",
        rf_generation_scope="future_only", rf_sigma_mode="const",
        rf_endpoint_mode="zero", rf_loss="x", rf_w="none",
        rf_mode="sde", rf_g="linear", rf_noise_scale=1.0,
        rf_standard_source=True, rf_transport_reduction="sum",
        rf_joint_eta_sigma=False, rf_uans=False, rf_uans_alpha=0.0,
        rf_ls_detach_target=True, rf_ls_freeze_after_pretrain=True,
        rf_ls_graph_mode="graph", rf_mu_clip=0.0,
        rf_mu_mode="residual", rf_ls_mu_scale=0.1,
        rf_ls_sigma_min=1e-3, rf_ls_sigma_max=5.0,
        rf_eval_variant="flow", rf_gaussian_ls_only=False,
        rf_min_residual_epochs=1, selection_metric="crps",
        interval_alpha=0.1, eval_clip_nonneg=False,
        rf_residual_calibration="none", rf_residual_calibration_factors="",
        wd=0.0, lr_scheduler="none", mask_ratio=0.0,
        eval_history_mask_ratio=0.0, eval_noise_std=0.0,
        lr=0.002, batch_size=8, hidden_size=32, rf_ls_hidden=128,
        T_h=12, T_p=12, rf_delta=1e-3, early_stop=10,
    )
    if not 1 <= int(params.get('epoch', 300)) <= 300:
        raise ValueError('The manuscript epoch budget must be between 1 and 300')
    if experiment == 'standard':
        if params.get('rf_eval_sigma') is not None or params.get('rf_eval_steps_grid'):
            raise ValueError('Inference sweeps require --paper_experiment sensitivity')
        params.update(rf_ls_pretrain_epochs=5, rf_steps=20, n_samples=100, rf_sigma=None)
    params.setdefault("rf_ls_pretrain_epochs", 5)
    if params.get("rf_sigma") is None:
        name = str(params.get("data", "PEMS08")).upper()
        if name in ("AIR_BJ", "AIR_GZ"):
            params["rf_sigma"] = 0.08
        elif name.startswith("PEMS"):
            params["rf_sigma"] = 0.05
        else:
            raise ValueError(f"The manuscript does not define gamma for {name}")
    if variant != "main":
        params["n_samples"] = 8
    if variant in ("joint_window_raw", "future_only_raw"):
        params.update(rf_target_space="raw", rf_ls_pretrain_epochs=0)
        if variant == "joint_window_raw":
            params["rf_generation_scope"] = "joint_history_future"
    elif variant == "mean_centered":
        params["rf_target_space"] = "mean_centered"
    elif variant == "no_graph":
        params["rf_ls_graph_mode"] = "none"
    elif variant == "no_freezing":
        # Joint training from epoch one: gradients of L_T + L_LS reach both
        # modules. The manuscript does not specify the auxiliary weight; use 1.
        params.update(rf_ls_pretrain_epochs=0,
                      rf_ls_freeze_after_pretrain=False,
                      rf_ls_detach_target=False, rf_ls_loss_w=1.0)
    return params


def checkpoint_protocol(config, clean_data):
    """Persist training/data semantics independently of inference overrides."""
    return {
        "version": PROTOCOL_VERSION,
        "dataset": config.data.name,
        "split": config.data.split_tag,
        "val_start_idx": int(config.data.val_start_idx),
        "test_start_idx": int(config.data.test_start_idx),
        "T_h": int(config.model.T_h), "T_p": int(config.model.T_p),
        "V": int(config.model.V), "F": int(config.model.F),
        "hidden_size": int(config.model.d_h),
        "mean": float(clean_data.mean), "std": float(clean_data.std),
        "graph_sha256": hashlib.sha256(clean_data.adj.tobytes()).hexdigest(),
        "features_sha256": hashlib.sha256(clean_data.feature.tobytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(clean_data.label.tobytes()).hexdigest(),
        "air_nan_fill": config.data.air_nan_fill,
        "missing_targets": "observed_entries_only",
        "crps": "empirical_cdf_integral",
        "paper_profile": config.model.paper_profile,
        "paper_variant": getattr(config, "paper_variant", "main"),
    }


def validate_checkpoint_protocol(model, expected):
    actual = getattr(model, "training_protocol", None)
    if actual is None:
        raise ValueError("Checkpoint has no training protocol; retrain with the fixed TeX protocol")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Checkpoint protocol mismatch for {key}: "
                             f"saved={actual.get(key)!r}, requested={value!r}")


def should_early_stop(epoch, best_epoch, patience):
    """Stop after exactly patience consecutive epochs without improvement."""
    return epoch - best_epoch >= patience

# NNI is optional. Some environments have incompatible scipy/nni wheels.
try:
    import nni  # type: ignore
except Exception:
    nni = None


def str2bool(v):
    """Argparse helper: correctly parse bool from CLI strings."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def str2intlist(v):
    """Parse comma-separated integers from CLI."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    text = str(v).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def str2floatlist(v):
    """Parse comma-separated floating-point values from CLI."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    text = str(v).strip()
    if not text:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def normalize_split_ratio(value):
    """Use the fixed 6:2:2 chronological split, including when omitted."""
    if value is None or str(value).strip() == "":
        return "622"
    text = str(value).strip().lower().replace(":", "").replace("-", "")
    if text != "622":
        raise ValueError(
            f"Unsupported split ratio '{value}'. Only 6:2:2 (or 622) is supported."
        )
    return "622"


def split_ratio_arg(value):
    try:
        return normalize_split_ratio(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def apply_paper_profile(params: dict) -> dict:
    """
    Freeze the canonical method variant used for paper writing while leaving
    dataset-specific optimization/noise-scale hyperparameters configurable.
    """
    profile = str(params.get('paper_profile', 'none'))
    if profile in ('www2027', 'canonical_ls_residual'):
        return apply_www2027_protocol(params)
    if params.get('rf_sigma') is None:
        params['rf_sigma'] = 0.08 if str(params.get('data', '')).startswith('AIR') else 0.05
    if profile == 'none':
        return params

    if profile == 'canonical_bridge':
        params.update({
            'engine': 'rf_sde_xpred',
            'rf_target_space': 'raw',
            'rf_generation_scope': 'future_only',
            'rf_sigma_mode': 'learned',
            'rf_endpoint_mode': 'zero',
            'rf_loss': 'x',
            'rf_w': 'none',
            'rf_mode': 'sde',
            'rf_g': 'linear',
            'rf_joint_eta_sigma': False,
        })
        return params

    if profile == 'canonical_bridge_uans':
        params.update({
            'engine': 'rf_sde_xpred',
            'rf_target_space': 'raw',
            'rf_generation_scope': 'future_only',
            'rf_sigma_mode': 'learned',
            'rf_endpoint_mode': 'zero',
            'rf_loss': 'x',
            'rf_w': 'none',
            'rf_mode': 'sde',
            'rf_g': 'linear',
            'rf_joint_eta_sigma': False,
            'rf_uans': True,
        })
        return params

    raise ValueError(f"Unknown paper profile: {profile}")

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--engine", type=str, default='rf_sde_xpred', choices=['diffstg', 'rf_sde_xpred'],
                        help="Model engine: baseline diffusion (diffstg) or linear-interp x-pred SDE (rf_sde_xpred)")
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=12)
    parser.add_argument("--T_p", type=int, default=None, help="Prediction horizon; default equals T_h for backward-compatibility")

    # rf_sde_xpred (engine-specific)
    parser.add_argument("--rf_steps", type=int, default=20, help="Number of EM steps for RF-SDE sampling")
    parser.add_argument("--rf_mode", type=str, default="sde", choices=["sde", "ode"], help="Sampling mode: sde (main) or ode (aux)")
    parser.add_argument("--rf_noise_scale", type=float, default=1.0, help="Initial noise scale for z0 ~ N(0, noise_scale^2 I)")
    parser.add_argument("--rf_sigma", type=float, default=None, help="Gamma: defaults to 0.08 for AIR and 0.05 for PEMS")
    parser.add_argument("--rf_standard_source", type=str2bool, default=True,
                        help="Use N(0,I) source independently of gamma for controlled ablations")
    parser.add_argument("--rf_transport_reduction", choices=['sum', 'mean'], default='sum',
                        help="Per-trajectory squared L2 sum, averaged over minibatch (Eq. 13)")
    parser.add_argument(
        "--rf_eval_sigma",
        type=float,
        default=None,
        help=(
            "Optional test-time override for RF-SDE diffusion sigma after "
            "loading a checkpoint. Leave unset to preserve the checkpoint value."
        ),
    )
    parser.add_argument("--rf_sigma_mode", type=str, default="const", choices=["const", "cond_hist_std", "learned"],
                        help="Non-stationary sigma: const (baseline), cond_hist_std (A1), learned (A2 sigma_net)")
    parser.add_argument("--rf_sigma_window", type=int, default=0,
                        help="A1: window size on history for std; 0 uses full T_h, >0 uses last window steps")
    parser.add_argument("--rf_sigma_min", type=float, default=1e-4, help="A1: clamp lower bound for conditional sigma")
    parser.add_argument("--rf_sigma_max", type=float, default=1.0, help="A1: clamp upper bound for conditional sigma")
    parser.add_argument("--rf_sigma_hidden", type=int, default=128, help="A2: hidden size for sigma_net")
    parser.add_argument("--rf_mu_hidden", type=int, default=128, help="LSNM endpoint: hidden size for mu_net")
    parser.add_argument("--rf_mu_mode", type=str, default="residual", choices=["abs", "residual"],
                        help="LSNM f(X) mode: abs predicts absolute mu, residual predicts baseline+scaled-delta")
    parser.add_argument("--rf_mu_scale", type=float, default=0.1,
                        help="Scale for residual mu branch when rf_mu_mode=residual")
    parser.add_argument("--rf_mu_clip", type=float, default=0.0,
                        help="Symmetric clamp for endpoint mean mu; <=0 disables clipping")
    parser.add_argument("--rf_mu_loss_w", type=float, default=0.1,
                        help="Weight of direct mu supervision loss: MSE(mu(X), future)")
    parser.add_argument("--rf_sigma_target_window", type=int, default=24, help="A2: trailing window length for sigma pseudo-label from (his+fut)")
    parser.add_argument("--rf_sigma_loss_w", type=float, default=0.1, help="A2: auxiliary loss weight for sigma regression")
    parser.add_argument("--rf_sigma_log_loss", type=str2bool, default=True, help="A2: regress sigma in log-space for stability")
    parser.add_argument("--rf_joint_eta_sigma", type=str2bool, default=False,
                        help="Step-3: jointly optimize eta_theta and sigma_theta via heteroscedastic main loss")
    parser.add_argument("--rf_joint_log_coeff", type=float, default=1.0,
                        help="Step-3: coefficient for log(sigma) term in joint heteroscedastic loss")
    parser.add_argument("--rf_endpoint_mode", type=str, default="zero", choices=["zero", "lsnm"],
                        help="Endpoint distribution mode: zero uses N(0, sigma), lsnm uses N(mu(X), sigma(X))")
    parser.add_argument("--rf_target_space", type=str, default="location_scale",
                        choices=["raw", "mean_centered", "location_scale"],
                        help="Flow target coordinates: raw future, mean-centered residual, or location-scale residual")
    parser.add_argument(
        "--rf_generation_scope",
        type=str,
        default="future_only",
        choices=["future_only", "joint_history_future"],
        help=(
            "Contribution-1 ablation: transport only the future target or the "
            "complete history+future sequence with the same RF-SDE"
        ),
    )
    parser.add_argument("--rf_ls_hidden", type=int, default=128,
                        help="Hidden size of the graph-aware conditional location-scale head")
    parser.add_argument(
        "--rf_ls_graph_mode",
        type=str,
        default="graph",
        choices=["graph", "none"],
        help=(
            "Graph ablation for the conditional location-scale head: graph "
            "uses bidirectional adjacency propagation; none uses identity "
            "propagation with the same projections and parameter count"
        ),
    )
    parser.add_argument("--rf_ls_mu_scale", type=float, default=0.1,
                        help="Scale of the learned correction around the persistence location")
    parser.add_argument("--rf_ls_sigma_min", type=float, default=1e-3,
                        help="Lower bound for the conditional residual scale")
    parser.add_argument("--rf_ls_sigma_max", type=float, default=5.0,
                        help="Upper bound for the conditional residual scale")
    parser.add_argument("--rf_ls_loss_w", type=float, default=1.0,
                        help="Weight of Gaussian NLL (or location MSE for mean-centered ablation)")
    parser.add_argument("--rf_ls_detach_target", type=str2bool, default=True,
                        help="Stop LS-head gradients through the flow target transform to prevent scale degeneracy")
    parser.add_argument("--rf_ls_warmup_epochs", type=int, default=0,
                        help="Deprecated first-generation warm-up option; use rf_ls_pretrain_epochs")
    parser.add_argument("--rf_ls_pretrain_epochs", type=int, default=5,
                        help="Stage 1 epochs that train only the conditional location-scale head")
    parser.add_argument("--rf_ls_freeze_after_pretrain", type=str2bool, default=True,
                        help="Freeze the location-scale coordinate system during Stage 2 flow training")
    parser.add_argument(
        "--rf_eval_variant",
        type=str,
        default="flow",
        choices=["flow", "gaussian_ls"],
        help="Evaluate the learned flow or the conditional Gaussian LS head alone",
    )
    parser.add_argument(
        "--rf_gaussian_ls_only",
        type=str2bool,
        default=False,
        help=(
            "Train only the graph-aware location-scale head, select it by "
            "validation Gaussian NLL, and skip Stage-2 Flow training"
        ),
    )
    parser.add_argument("--rf_g", type=str, default='linear', choices=['linear', 'square'], help="Pinning noise schedule: linear=(1-t), square=(1-t^2)")
    parser.add_argument("--rf_uans", type=str2bool, default=False,
                        help="Enable uncertainty-aware noise schedule (UANS) on g(t)")
    parser.add_argument("--rf_uans_alpha", type=float, default=0.0,
                        help="UANS strength; >0 gives slower decay for high-uncertainty regions")
    parser.add_argument("--rf_uans_min_pow", type=float, default=0.5,
                        help="UANS lower bound for decay power")
    parser.add_argument("--rf_uans_max_pow", type=float, default=2.0,
                        help="UANS upper bound for decay power")
    parser.add_argument("--rf_uans_floor", type=float, default=0.0,
                        help="UANS floor for schedule value in [0,1)")
    parser.add_argument("--rf_delta", type=float, default=1e-3, help="Numerical stabilizer for (1-t+delta) denominator")
    parser.add_argument("--rf_loss", type=str, default='x', choices=['x', 'v'], help="Training loss type: x-pred MSE or v-pred MSE (JiT-style)")
    parser.add_argument("--rf_w", type=str, default='none', choices=['none', 'jit'], help="Time weighting for x-loss: none (w=1) or jit (w=1/(1-t)^2)")
    parser.add_argument("--rf_sample_chunk", type=int, default=10, help="Chunk size for multi-sample generation to avoid OOM (e.g., 5/10/20)")
    parser.add_argument("--paper_profile", type=str, default=None,
                        choices=['www2027', 'none', 'canonical_bridge', 'canonical_bridge_uans', 'canonical_ls_residual'],
                        help="Freeze a paper-facing canonical method variant")
    parser.add_argument("--paper_variant", choices=PAPER_VARIANTS, default='main',
                        help="WWW2027 main method or a controlled component ablation")
    parser.add_argument("--paper_experiment", choices=['standard', 'sensitivity', 'diagnostic'], default='standard',
                        help="Standard TeX settings, numerical sensitivity overrides, or a synthetic diagnostic")

    # eval
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--val_n_samples', type=int, default=8,
                        help="Number of trajectories used for validation/checkpoint selection")
    parser.add_argument('--eval_warmup_batches', type=int, default=0,
                        help="Untimed test warm-up batches for inference efficiency measurement")
    parser.add_argument('--selection_metric', type=str, default='crps',
                        choices=['mae', 'crps'],
                        help="Validation metric used by checkpointing, LR scheduling, and early stopping")
    parser.add_argument("--rf_eval_steps_grid", type=str, default="",
                        help="Optional comma-separated RF sampling steps for test-time efficiency sweeps")
    parser.add_argument("--eval_history_mask_ratio", type=float, default=0.0,
                        help="Randomly mask a ratio of history values during validation/test for robustness evaluation")
    parser.add_argument("--eval_noise_std", type=float, default=0.0,
                        help="Add Gaussian noise with this std to history during validation/test")
    parser.add_argument("--forecast_save_limit", type=int, default=50,
                        help="Number of test instances to save into forecast pkl; <=0 saves all")
    parser.add_argument(
        "--eval_window_limit",
        type=int,
        default=0,
        help=(
            "Maximum number of test windows to evaluate; <=0 evaluates the "
            "entire test split. Unlike forecast_save_limit, this reduces inference."
        ),
    )
    parser.add_argument(
        "--detailed_eval",
        type=str2bool,
        default=False,
        help="Stream horizon, multi-coverage, case-bootstrap, and residual diagnostics",
    )
    parser.add_argument(
        "--detailed_coverage_levels",
        type=str,
        default="0.5,0.8,0.9,0.95",
        help="Comma-separated nominal coverages recorded by detailed evaluation",
    )
    parser.add_argument("--eval_clip_nonneg", type=str2bool, default=False,
                        help="Clip validation/test predictions to be nonnegative after denormalization")
    parser.add_argument("--interval_alpha", type=float, default=0.1,
                        help="Central interval error rate; use 0.1 for PICP90")
    parser.add_argument(
        "--rf_residual_calibration",
        type=str,
        default="none",
        choices=["none", "global", "horizon"],
        help=(
            "Fit validation-only ensemble spread scaling for residual flows; "
            "horizon learns one factor per prediction step"
        ),
    )
    parser.add_argument(
        "--rf_residual_calibration_samples",
        type=int,
        default=0,
        help="Validation trajectories used to fit spread calibration; <=0 uses n_samples",
    )
    parser.add_argument("--rf_residual_calibration_min", type=float, default=0.5,
                        help="Lower clamp for fitted residual spread factors")
    parser.add_argument("--rf_residual_calibration_max", type=float, default=4.0,
                        help="Upper clamp for fitted residual spread factors")
    parser.add_argument(
        "--rf_residual_calibration_factors",
        type=str,
        default="",
        help=(
            "Optional comma-separated pre-fitted spread factors. Providing "
            "these skips validation fitting; use one global or T_p horizon factors"
        ),
    )

    # train
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto | cuda | cuda:0 | cuda:1 | cpu")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed for training/evaluation reproducibility")
    parser.add_argument("--is_train", type=str2bool, default=True) # train or evaluate
    parser.add_argument("--data", type=str, default='PEMS08')
    parser.add_argument(
        "--split_ratio",
        type=split_ratio_arg,
        default="622",
        metavar="6:2:2",
        help=(
            "Chronological train/validation/test split fixed to 6:2:2 for all "
            "datasets, including PEMS08. Optional; only 6:2:2/622 is supported."
        ),
    )
    parser.add_argument("--air_nan_fill", type=str, default='ffill', choices=['zero', 'node_mean', 'ffill'],
                        help="Missing-value fill strategy for AIR datasets")
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=str2bool, default=False)
    parser.add_argument("--nni", type=str2bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Optional checkpoint path (.rfstg/.pth) for evaluation without training")
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Adam weight decay; fixed manuscript uses ordinary Adam")
    parser.add_argument("--lr_scheduler", choices=['none', 'plateau'], default='none')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--rf_min_residual_epochs", type=int, default=1,
                        help="Minimum residual-flow epochs before a non-raw checkpoint is eligible")
    parser.add_argument("--eval_tag", type=str, default="",
                        help="Optional suffix for eval-only logs/forecasts (e.g. no_uans or uans)")
    parser.add_argument("--result_json", type=str, default="",
                        help="Optional path for machine-readable test metrics")

    args, _ = parser.parse_known_args()
    if args.paper_profile is None:
        args.paper_profile = 'none' if args.engine == 'diffstg' else 'www2027'
    return args


def infer_dataset_meta(data_name: str, feature_file: str):
    """
    Infer generic dataset metadata from flow.npy when available so that new
    PEMS-style datasets can be plugged in with minimal code changes.
    """
    meta = {}
    if os.path.exists(feature_file):
        arr = np.load(feature_file, mmap_mode='r')
        if arr.ndim == 3:
            meta['num_features'] = 1 if ('PEMS' in data_name or 'AIR' in data_name or 'Metro' in data_name) else int(arr.shape[2])
            meta['num_vertices'] = int(arr.shape[1])
            meta['total_steps'] = int(arr.shape[0])
        elif arr.ndim == 2:
            meta['num_features'] = 1
            meta['num_vertices'] = int(arr.shape[1])
            meta['total_steps'] = int(arr.shape[0])
    name_upper = str(data_name).upper()
    if 'PEMS' in name_upper:
        meta.setdefault('points_per_hour', 12)
        meta.setdefault('steps_per_day', 288)
    elif 'METR' in name_upper or 'METRO' in name_upper:
        # METR-LA uses 5-min granularity.
        meta.setdefault('points_per_hour', 12)
        meta.setdefault('steps_per_day', 288)
    elif 'KNOWAIR' in name_upper:
        # Official KnowAir is sampled every 3 hours, i.e. 8 steps per day.
        meta.setdefault('points_per_hour', 1)
        meta.setdefault('steps_per_day', 8)
    elif 'AIR' in name_upper:
        meta.setdefault('points_per_hour', 1)
        meta.setdefault('steps_per_day', 24)
    else:
        meta.setdefault('points_per_hour', 1)
        meta.setdefault('steps_per_day', 24)
    return meta

def default_config(data='AIR_BJ', preferred_device: str = "auto", split_ratio=None):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1
    inferred = infer_dataset_meta(config.data.name, config.data.feature_file)

    # Backward-compatible fallbacks for datasets that might not exist yet on disk.
    fallback_meta = {
        'PEMS08': {'num_features': 1, 'num_vertices': 170, 'total_steps': 17856, 'points_per_hour': 12, 'steps_per_day': 288},
        'AIR_BJ': {'num_features': 1, 'num_vertices': 34, 'total_steps': 8760, 'points_per_hour': 1, 'steps_per_day': 24},
        'AIR_GZ': {'num_features': 1, 'num_vertices': 41, 'total_steps': 8760, 'points_per_hour': 1, 'steps_per_day': 24},
        'PEMS04': {'num_features': 1, 'num_vertices': 307, 'total_steps': 16992, 'points_per_hour': 12, 'steps_per_day': 288},
        'PEMS07': {'num_features': 1, 'num_vertices': 883, 'total_steps': 28224, 'points_per_hour': 12, 'steps_per_day': 288},
        'PEMS-BAY': {'num_features': 1, 'num_vertices': 325, 'total_steps': 52116, 'points_per_hour': 12, 'steps_per_day': 288},
        'METR-LA': {'num_features': 1, 'num_vertices': 207, 'total_steps': 34272, 'points_per_hour': 12, 'steps_per_day': 288},
        'KnowAir': {'num_features': 1, 'num_vertices': 184, 'total_steps': 11688, 'points_per_hour': 1, 'steps_per_day': 8},
    }.get(config.data.name, {})

    dataset_meta = {}
    dataset_meta.update(fallback_meta)
    dataset_meta.update(inferred)

    if not dataset_meta:
        raise ValueError(
            f"Unable to infer dataset meta for '{config.data.name}'. "
            "Please provide a dataset with flow.npy or add a fallback entry."
        )

    config.data.num_features = int(dataset_meta['num_features'])
    config.data.num_vertices = int(dataset_meta['num_vertices'])
    config.data.points_per_hour = int(dataset_meta['points_per_hour'])
    config.data.steps_per_day = int(dataset_meta.get('steps_per_day', config.data.points_per_hour * 24))
    total_steps = int(dataset_meta['total_steps'])

    # Establish timestamp boundaries before CleanDataset fits normalization.
    # All datasets use the same fixed 6:2:2 chronological protocol.
    split_tag = normalize_split_ratio(split_ratio)
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    config.data.train_ratio = train_ratio
    config.data.val_ratio = val_ratio
    config.data.test_ratio = test_ratio
    config.data.split_tag = split_tag
    config.data.val_start_idx = int(total_steps * train_ratio)
    config.data.test_start_idx = int(total_steps * (train_ratio + val_ratio))

    preferred_device = str(preferred_device).strip().lower()
    if preferred_device != "auto":
        if preferred_device.startswith("cuda"):
            if torch.cuda.is_available():
                # Support both "cuda" and "cuda:idx"
                if ":" in preferred_device:
                    torch.cuda.set_device(preferred_device)
                    device = torch.device(preferred_device)
                else:
                    torch.cuda.set_device("cuda:0")
                    device = torch.device("cuda:0")
                config.gpu_id = None
            else:
                print(f"[warn] CUDA not available, fallback to CPU (requested: {preferred_device})")
                config.gpu_id = None
                device = torch.device('cpu')
        else:
            config.gpu_id = None
            device = torch.device('cpu')
    else:
        gpu_id = GPU().get_usefuel_gpu(max_memory=6000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
        config.gpu_id = gpu_id
        if gpu_id is not None:
            cuda_id = GpuId2CudaId(gpu_id)
            torch.cuda.set_device(f"cuda:{cuda_id}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    config.model = edict()

    config.model.T_p = 12
    config.model.T_h = 12
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.steps_per_day
    config.model.device = device
    config.model.d_h = 32

    # config for diffusion model
    config.model.N = 200
    config.model.sample_steps = 200
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    config.n_samples = 100

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False  # Whether run the code in the test mode
    config.epoch = 300  # Number of max training epoch
    config.optimizer = "adam"
    config.lr = 0.002
    config.batch_size = 8
    config.wd = 0.0
    config.early_stop = 10
    config.start_epoch = 0
    config.device = device
    config.logger = Logger()


    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)
    return config

def evals(
    model,
    data_loader,
    epoch,
    metric,
    config,
    clean_data,
    mode='Test',
    seed: int = 2022,
    selection_eligible: bool = True,
):
    setup_seed(int(seed))

    time_lst = []
    interval_alpha = float(getattr(config, 'interval_alpha', 0.1))
    detailed_eval = bool(getattr(config, 'detailed_eval', False)) and mode.lower() == 'test'
    coverage_levels = getattr(config, 'detailed_coverage_levels', [0.5, 0.8, 0.9, 0.95])
    metrics_future_acc = StreamingMetricAccumulator(
        alpha=interval_alpha,
        detailed=detailed_eval,
        coverage_levels=coverage_levels,
        store_case_metrics=detailed_eval,
    )
    metrics_history_acc = StreamingMetricAccumulator(alpha=interval_alpha)
    location_diagnostic_acc = None
    if detailed_eval and hasattr(model, '_target_space') and model._target_space() != 'raw':
        location_diagnostic_acc = ConditionalResidualDiagnosticAccumulator()
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    if mode.lower() == 'test':
        with torch.no_grad():
            for warmup_index, batch in enumerate(data_loader):
                if warmup_index >= int(getattr(config, 'eval_warmup_batches', 0)):
                    break
                future, history, pos_w, pos_d = to_device(batch, config.device)
                condition = torch.cat((history, torch.zeros_like(future)), dim=1).transpose(1, 3)
                model((condition, pos_w, pos_d), resolve_eval_n_samples(config, mode))
        # Warm-up must not change the generated evaluation ensemble.
        setup_seed(int(seed))

    # CUDA kernels are asynchronous.  Synchronizing at the timing boundaries
    # keeps inference times comparable across samplers, while resetting the
    # allocator statistics here reports the peak of the test pass (including
    # the already-loaded model), not an unrelated training/validation peak.
    eval_device = torch.device(config.device)
    measure_cuda = eval_device.type == 'cuda' and torch.cuda.is_available()
    if measure_cuda:
        torch.cuda.synchronize(eval_device)
        torch.cuda.reset_peak_memory_stats(eval_device)

    samples, targets = [], []
    saved_count = 0
    save_limit = int(getattr(config, 'forecast_save_limit', 50)) if mode == 'test' else 0
    eval_window_limit = (
        int(getattr(config, 'eval_window_limit', 0))
        if mode.lower() == 'test'
        else 0
    )
    evaluated_count = 0
    for i, batch in enumerate(data_loader):
        if i > 0 and config.is_test: break
        if eval_window_limit > 0:
            remaining = eval_window_limit - evaluated_count
            if remaining <= 0:
                break
            batch_size = int(batch[0].shape[0])
            if batch_size > remaining:
                batch = tuple(value[:remaining] for value in batch)
        if measure_cuda:
            torch.cuda.synchronize(eval_device)
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)

        history_eval = history.clone()
        if mode.lower() in ('val', 'test'):
            if getattr(config, 'eval_history_mask_ratio', 0.0) > 0:
                mask = torch.rand_like(history_eval) < float(config.eval_history_mask_ratio)
                history_eval[mask] = 0
            if getattr(config, 'eval_noise_std', 0.0) > 0:
                history_eval = history_eval + torch.randn_like(history_eval) * float(config.eval_noise_std)

        x = torch.cat((history, future), dim=1).to(config.device)  # in cpu (B, T, V, F), T =  T_h + T_p
        x_masked = torch.cat((history_eval, torch.zeros_like(future)), dim=1).to(config.device)  # (B, T, V, F)
        target_save = x.detach().cpu()
        evaluated_count += int(target_save.shape[0])
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

        n_samples = resolve_eval_n_samples(config, mode)
        x_hat = model((x_masked, pos_w, pos_d), n_samples) # (B, n_samples, F, V, T)
        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)
        # assert x.shape == x_hat.shape, f"shape of x ({x.shape}) does not equal to shape of x_hat ({x_hat.shape})"

        calibration_factors = getattr(config, 'rf_residual_calibration_factors', [])
        if calibration_factors:
            # Apply only to the generated future.  Scaling is affine-invariant,
            # so doing this before reverse normalization is equivalent to
            # observation-space calibration and also makes saved forecasts
            # reflect the calibrated predictive distribution.
            future_samples = x_hat[..., -config.model.T_p:].transpose(2, 4)
            future_samples = apply_residual_spread_calibration(
                future_samples, calibration_factors
            )
            x_hat[..., -config.model.T_p:] = future_samples.transpose(2, 4)

        sample_save = x_hat.transpose(2,4).cpu()
        if measure_cuda:
            torch.cuda.synchronize(eval_device)
        if mode == 'test':
            if save_limit <= 0:
                samples.append(sample_save)
                targets.append(target_save)
            elif saved_count < save_limit:
                take = min(save_limit - saved_count, target_save.shape[0])
                if take > 0:
                    samples.append(sample_save[:take])
                    targets.append(target_save[:take])
                    saved_count += take

        time_lst.append((timer() - time_start))
        x, x_hat= clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:] # future

        _y_true_ = f_x.transpose(1, 3).detach().cpu()  # y_true: (B, T_p, V, D)
        _y_pred_ = f_x_hat.transpose(2, 4).detach().cpu() # y_pred: (B, n_samples, T_p, V, D)
        if getattr(config, 'eval_clip_nonneg', False):
            _y_pred_ = torch.clamp(_y_pred_, min=0.0)
        metrics_future_acc.update(_y_true_, _y_pred_)

        if location_diagnostic_acc is not None:
            history_for_location = history_eval.transpose(1, 3)
            mu_normalized, _sigma_normalized = model._compute_location_scale(
                history_for_location
            )
            mu_observation = clean_data.reverse_normalization(mu_normalized)
            mu_observation = mu_observation.transpose(1, 3).detach().cpu()
            location_diagnostic_acc.update(_y_true_, mu_observation)

        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).detach().cpu()  # y_true: (B, T_p, V, D)
        _y_pred_ = h_x_hat.transpose(2, 4).detach().cpu()
        if getattr(config, 'eval_clip_nonneg', False):
            _y_pred_ = torch.clamp(_y_pred_, min=0.0)
        metrics_history_acc.update(_y_true_, _y_pred_)

    time_cost = np.sum(time_lst)
    metric.metrics = metrics_future_acc.compute_metrics()
    metric.diagnostics = metrics_future_acc.compute_diagnostics()
    if location_diagnostic_acc is not None:
        metric.diagnostics['conditional_residual'] = location_diagnostic_acc.compute()
    metrics_history.metrics = metrics_history_acc.compute_metrics()
    selection_metric = str(getattr(config, 'selection_metric', 'mae')).lower()
    metric.update_best_metrics(
        epoch=epoch,
        selection_metric=selection_metric,
        eligible=selection_eligible,
    )
    metric.metrics['time'] = time_cost
    metric.metrics['evaluated_windows'] = int(evaluated_count)
    metric.metrics['evaluated_trajectories'] = int(evaluated_count * resolve_eval_n_samples(config, mode))
    metric.metrics['ms_per_trajectory'] = float(time_cost * 1000 / max(metric.metrics['evaluated_trajectories'], 1))
    metrics_history.metrics['time'] = time_cost
    if measure_cuda:
        mib = 1024.0 ** 2
        metric.metrics['peak_gpu_memory_allocated_mb'] = (
            torch.cuda.max_memory_allocated(eval_device) / mib
        )
        metric.metrics['peak_gpu_memory_reserved_mb'] = (
            torch.cuda.max_memory_reserved(eval_device) / mib
        )

    if mode == 'test': # save the prediction result to file
        if samples:
            samples = torch.cat(samples, dim=0)
            targets = torch.cat(targets, dim=0)
        else:
            samples = torch.empty((0, config.n_samples, config.model.T_h + config.model.T_p, config.model.V, config.model.F))
            targets = torch.empty((0, config.model.T_h + config.model.T_p, config.model.V, config.model.F))
        observed_flag = torch.isfinite(targets).to(targets.dtype)
        evaluate_flag = torch.zeros_like(observed_flag)
        evaluate_flag[:, -config.model.T_p:, :, :] = observed_flag[:, -config.model.T_p:, :, :]
        import pickle
        with open (config.forecast_path, 'wb') as f:
            pickle.dump([samples, targets, observed_flag, evaluate_flag], f)
        with open(config.forecast_path + '.protocol.json', 'w', encoding='utf-8') as file_obj:
            json.dump({
                'training_protocol': model.training_protocol,
                'paper_variant': getattr(config, 'paper_variant', 'main'),
                'paper_experiment': config.paper_experiment,
                'engine': config.engine,
                'checkpoint': config.ckpt_path or config.model_path,
                'checkpoint_selection_metric': getattr(model, 'checkpoint_selection_metric', 'unknown'),
                'target_space': model._target_space() if hasattr(model, '_target_space') else 'diffusion',
                'n_samples': resolve_eval_n_samples(config, mode),
                'nominal_coverage': 1.0 - config.interval_alpha,
                'calibration': getattr(config, 'rf_residual_calibration', 'none'),
                'eval_clip_nonneg': getattr(config, 'eval_clip_nonneg', False),
                'units': 'training_standardized',
            }, file_obj, indent=2)

        message = f"predict_path = '{config.forecast_path}'"
        config.logger.message_buffer += f"{message}\n"
        config.logger.write_message_buffer()


    if config.nni and (nni is not None):
        nni.report_intermediate_result(metric.metrics[selection_metric])

    # log of performance in future prediction
    if metric.last_improved.get(selection_metric, False):
        message = f" |[{metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}]"
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    if mode.lower() == 'val':
        eligibility = '' if selection_eligible else ' warmup'
        message += (
            f" CRPS {metric.metrics['crps']:.4f}"
            f" PICP {metric.metrics['picp']:.4f}"
            f" MIS {metric.metrics['mis']:.2f}"
            f" Width {metric.metrics['interval_width']:.2f}"
            f" select={selection_metric}{eligibility}"
        )
    print(message, end='', flush=False)
    config.logger.message_buffer += message

    # log of performance in historical prediction
    message = f" | {metrics_history.metrics['mae']:<7.2f}{metrics_history.metrics['rmse']:<7.2f}{time_cost:<5.2f}s"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += f"{message}\n"

    # write log message buffer
    config.logger.write_message_buffer()

    torch.cuda.empty_cache()
    return metric


@torch.no_grad()
def fit_residual_spread_calibration(
    model,
    data_loader,
    config,
    clean_data,
    seed: int,
):
    """Fit residual ensemble spread factors using validation targets only."""
    mode = str(getattr(config, 'rf_residual_calibration', 'none')).lower()
    if mode == 'none':
        return []

    setup_seed(int(seed))
    n_samples = int(getattr(config, 'rf_residual_calibration_samples', 0))
    if n_samples <= 0:
        n_samples = int(config.n_samples)
    if n_samples < 2:
        raise ValueError('Residual spread calibration requires at least 2 samples')

    calibrator = CoverageSpreadCalibrator(
        alpha=float(getattr(config, 'interval_alpha', 0.1)),
        mode=mode,
        min_factor=float(getattr(config, 'rf_residual_calibration_min', 0.5)),
        max_factor=float(getattr(config, 'rf_residual_calibration_max', 4.0)),
    )
    model.eval()
    for batch in data_loader:
        future, history, pos_w, pos_d = to_device(batch, config.device)
        x = torch.cat((history, future), dim=1).to(config.device)
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
        x = x.transpose(1, 3)
        x_masked = x_masked.transpose(1, 3)

        x_hat = model((x_masked, pos_w, pos_d), n_samples)
        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p):
            x_hat = x_hat.transpose(2, 4)
        x = clean_data.reverse_normalization(x)
        x_hat = clean_data.reverse_normalization(x_hat)
        future_true = x[:, :, :, -config.model.T_p:].transpose(1, 3).detach().cpu()
        future_pred = x_hat[:, :, :, :, -config.model.T_p:].transpose(2, 4).detach().cpu()
        if getattr(config, 'eval_clip_nonneg', False):
            future_pred = torch.clamp(future_pred, min=0.0)
        calibrator.update(future_true, future_pred)

    factors = calibrator.compute_factors()
    torch.cuda.empty_cache()
    return factors


@torch.no_grad()
def eval_location_scale_head(model, data_loader, config, clean_data):
    """Fast Stage-1 validation without running the generative sampler."""
    model.eval()
    count = 0
    abs_error_sum = 0.0
    objective_sum = 0.0
    residual_sum = 0.0
    residual_sq_sum = 0.0
    sigma_sum = 0.0

    target_space = model._target_space()
    for i, batch in enumerate(data_loader):
        if i > 0 and config.is_test:
            break
        future, history, _pos_w, _pos_d = to_device(batch, config.device)
        future = future.transpose(1, 3)  # (B,F,V,T_p)
        history = history.transpose(1, 3)  # (B,F,V,T_h)
        mu, sigma = model._compute_location_scale(history)

        valid = torch.isfinite(future)
        error = torch.where(valid, future - mu, torch.zeros_like(mu))
        residual = error / sigma.clamp_min(float(model.rf_cfg.ls_sigma_min))
        objective = 0.5 * residual.square() + torch.log(sigma.clamp_min(1e-8))

        n_values = int(valid.sum())
        count += n_values
        abs_error_sum += float(error.abs()[valid].sum())
        objective_sum += float(objective[valid].sum())
        residual_sum += float(residual[valid].sum())
        residual_sq_sum += float(residual.square()[valid].sum())
        sigma_sum += float(sigma[valid].sum())

    if count == 0:
        raise RuntimeError('Empty validation loader during location-scale pretraining')
    residual_mean = residual_sum / count
    residual_var = max(residual_sq_sum / count - residual_mean * residual_mean, 0.0)
    data_std = float(clean_data.std)
    return {
        'objective': objective_sum / count,
        'mu_mae': abs_error_sum / count * data_std,
        'sigma_mean': sigma_sum / count * data_std,
        'residual_mean': residual_mean,
        'residual_std': residual_var ** 0.5,
    }


from pprint import  pprint
def main(params: dict):
    params = apply_paper_profile(dict(params))
    # global seed control
    setup_seed(int(params.get('seed', 2022)))
    torch.set_num_threads(2)
    config = default_config(
        params['data'],
        preferred_device=params.get('device', 'auto'),
        split_ratio=params.get('split_ratio'),
    )
    config.seed = int(params.get('seed', 2022))
    config.paper_variant = params.get('paper_variant', 'main')
    config.paper_experiment = params.get('paper_experiment', 'standard')
    config.lr_scheduler = params.get('lr_scheduler', 'none')

    config.is_train = bool(params['is_train'])
    config.is_test = params['is_test']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.wd = float(params.get('wd', 0.0))
    config.epoch = int(params.get('epoch', 300))
    config.early_stop = int(params.get('early_stop', 10))
    config.mask_ratio = params['mask_ratio']
    config.engine = params.get('engine', 'diffstg')
    config.data.air_nan_fill = params.get('air_nan_fill', 'ffill')

    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    tp = params.get('T_p', None)
    if tp is None:
        tp = params['T_h']
    config.T_p = config.model.T_p = int(tp)
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']
    config.val_n_samples = max(1, int(params.get('val_n_samples', 8)))
    config.eval_warmup_batches = int(params.get('eval_warmup_batches', 0))
    config.selection_metric = str(params.get('selection_metric', 'crps')).lower()
    config.rf_min_residual_epochs = max(0, int(params.get('rf_min_residual_epochs', 1)))

    # rf_sde_xpred params
    config.model.rf_steps = params.get('rf_steps', 20)
    config.model.rf_mode = params.get('rf_mode', 'sde')
    config.model.rf_noise_scale = params.get('rf_noise_scale', 1.0)
    config.model.rf_standard_source = params.get('rf_standard_source', True)
    config.model.rf_transport_reduction = params.get('rf_transport_reduction', 'sum')
    config.model.rf_sigma = params.get('rf_sigma', 0.1)
    config.rf_eval_sigma = params.get('rf_eval_sigma', None)
    config.model.rf_sigma_mode = params.get('rf_sigma_mode', 'const')
    config.model.rf_sigma_window = params.get('rf_sigma_window', 0)
    config.model.rf_sigma_min = params.get('rf_sigma_min', 1e-4)
    config.model.rf_sigma_max = params.get('rf_sigma_max', 1.0)
    config.model.rf_sigma_hidden = params.get('rf_sigma_hidden', 128)
    config.model.rf_mu_hidden = params.get('rf_mu_hidden', 128)
    config.model.rf_mu_mode = params.get('rf_mu_mode', 'residual')
    config.model.rf_mu_scale = params.get('rf_mu_scale', 0.1)
    config.model.rf_mu_clip = params.get('rf_mu_clip', 0.0)
    config.model.rf_mu_loss_w = params.get('rf_mu_loss_w', 0.1)
    config.model.rf_sigma_target_window = params.get('rf_sigma_target_window', 24)
    config.model.rf_sigma_loss_w = params.get('rf_sigma_loss_w', 0.1)
    config.model.rf_sigma_log_loss = params.get('rf_sigma_log_loss', True)
    config.model.rf_joint_eta_sigma = params.get('rf_joint_eta_sigma', False)
    config.model.rf_joint_log_coeff = params.get('rf_joint_log_coeff', 1.0)
    config.model.rf_endpoint_mode = params.get('rf_endpoint_mode', 'zero')
    config.model.rf_target_space = params.get('rf_target_space', 'raw')
    config.model.rf_generation_scope = params.get(
        'rf_generation_scope', 'future_only'
    )
    config.model.rf_ls_hidden = params.get('rf_ls_hidden', 128)
    config.model.rf_ls_graph_mode = params.get('rf_ls_graph_mode', 'graph')
    config.model.rf_ls_mu_scale = params.get('rf_ls_mu_scale', 0.1)
    config.model.rf_ls_sigma_min = params.get('rf_ls_sigma_min', 1e-3)
    config.model.rf_ls_sigma_max = params.get('rf_ls_sigma_max', 5.0)
    config.model.rf_ls_loss_w = params.get('rf_ls_loss_w', 1.0)
    config.model.rf_ls_detach_target = params.get('rf_ls_detach_target', True)
    config.model.rf_ls_warmup_epochs = params.get('rf_ls_warmup_epochs', 0)
    config.model.rf_ls_pretrain_epochs = params.get('rf_ls_pretrain_epochs', 5)
    config.model.rf_ls_freeze_after_pretrain = params.get('rf_ls_freeze_after_pretrain', True)
    config.model.rf_eval_variant = params.get('rf_eval_variant', 'flow')
    config.rf_gaussian_ls_only = bool(params.get('rf_gaussian_ls_only', False))
    config.model.rf_g = params.get('rf_g', 'linear')
    config.model.rf_uans = params.get('rf_uans', False)
    config.model.rf_uans_alpha = params.get('rf_uans_alpha', 0.0)
    config.model.rf_uans_min_pow = params.get('rf_uans_min_pow', 0.5)
    config.model.rf_uans_max_pow = params.get('rf_uans_max_pow', 2.0)
    config.model.rf_uans_floor = params.get('rf_uans_floor', 0.0)
    config.model.rf_delta = params.get('rf_delta', 1e-3)
    config.model.rf_loss = params.get('rf_loss', 'x')
    config.model.rf_w = params.get('rf_w', 'none')
    config.model.rf_sample_chunk = params.get('rf_sample_chunk', 10)
    config.model.paper_profile = params.get('paper_profile', 'none')
    config.eval_history_mask_ratio = float(params.get('eval_history_mask_ratio', 0.0))
    config.eval_noise_std = float(params.get('eval_noise_std', 0.0))
    config.eval_clip_nonneg = bool(params.get('eval_clip_nonneg', False))
    config.interval_alpha = float(params.get('interval_alpha', 0.1))
    config.rf_residual_calibration = str(
        params.get('rf_residual_calibration', 'none')
    ).lower()
    config.rf_residual_calibration_samples = int(
        params.get('rf_residual_calibration_samples', 0)
    )
    config.rf_residual_calibration_min = float(
        params.get('rf_residual_calibration_min', 0.5)
    )
    config.rf_residual_calibration_max = float(
        params.get('rf_residual_calibration_max', 4.0)
    )
    config.rf_residual_calibration_factors = str2floatlist(
        params.get('rf_residual_calibration_factors', '')
    )
    config.forecast_save_limit = int(params.get('forecast_save_limit', 50))
    config.eval_window_limit = int(params.get('eval_window_limit', 0))
    if config.eval_window_limit < 0:
        raise ValueError('eval_window_limit must be non-negative')
    config.detailed_eval = bool(params.get('detailed_eval', False))
    config.detailed_coverage_levels = str2floatlist(
        params.get('detailed_coverage_levels', '0.5,0.8,0.9,0.95')
    )
    config.rf_eval_steps_grid = str2intlist(params.get('rf_eval_steps_grid', ''))
    config.eval_tag = str(params.get('eval_tag', '') or '').strip()
    config.result_json = str(params.get('result_json', '') or '').strip()

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        if config.nni and (nni is not None):
            # nni.report_intermediate_result(50)
            nni.report_final_result(50)
        return 0


    ckpt_path = str(params.get('ckpt_path', '') or '').strip()
    if ckpt_path:
        ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
    config.ckpt_path = ckpt_path

    # Avoid overlong filenames when many CLI params are enabled.
    # If ckpt_path is provided for eval-only, prefer using the checkpoint filename for logging/forecasting.
    if config.ckpt_path and (not bool(params.get('is_train', True))):
        ckpt_tag = os.path.splitext(os.path.basename(config.ckpt_path))[0]
        eval_tag = str(params.get('eval_tag', '') or '').strip()
        config.trial_name = f"{ckpt_tag}+eval-{eval_tag}" if eval_tag else ckpt_tag
    else:
        trial_name_raw = '+'.join([f"{v}" for k, v in params.items()])
        if len(trial_name_raw) > 120:
            digest = hashlib.md5(trial_name_raw.encode("utf-8")).hexdigest()[:10]
            config.trial_name = f"{params.get('engine', 'eng')}+{params.get('data', 'data')}+s{params.get('seed', 0)}+h{digest}"
        else:
            config.trial_name = trial_name_raw
        # Make every training split explicit so checkpoints from different
        # chronological protocols can never collide.
        config.trial_name += f"+split{config.data.split_tag}"
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    #  data pre-processing
    # print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    if config.engine == 'diffstg':
        model = DiffSTG(config.model)
    elif config.engine == 'rf_sde_xpred':
        # Map CLI args to RF-SDE config used in algorithm/rf_sde/model.py
        from algorithm.rf_sde.model import RFSDEConfig
        rf_cfg = RFSDEConfig(
            num_sampling_steps=int(getattr(config.model, 'rf_steps', 20)),
            sampling_mode=str(getattr(config.model, 'rf_mode', 'sde')),
            noise_scale=float(getattr(config.model, 'rf_noise_scale', 1.0)),
            standard_source=bool(config.model.rf_standard_source),
            transport_reduction=str(config.model.rf_transport_reduction),
            sigma=float(getattr(config.model, 'rf_sigma', 0.1)),
            g_schedule='square' if getattr(config.model, 'rf_g', 'linear') == 'square' else 'linear',
            sigma_mode=str(getattr(config.model, 'rf_sigma_mode', 'const')),
            sigma_window=int(getattr(config.model, 'rf_sigma_window', 0)),
            sigma_min=float(getattr(config.model, 'rf_sigma_min', 1e-4)),
            sigma_max=float(getattr(config.model, 'rf_sigma_max', 1.0)),
            sigma_hidden=int(getattr(config.model, 'rf_sigma_hidden', 128)),
            mu_hidden=int(getattr(config.model, 'rf_mu_hidden', 128)),
            mu_mode=str(getattr(config.model, 'rf_mu_mode', 'residual')),
            mu_scale=float(getattr(config.model, 'rf_mu_scale', 0.1)),
            mu_clip=float(getattr(config.model, 'rf_mu_clip', 0.0)),
            mu_loss_weight=float(getattr(config.model, 'rf_mu_loss_w', 0.1)),
            sigma_target_window=int(getattr(config.model, 'rf_sigma_target_window', 24)),
            sigma_loss_weight=float(getattr(config.model, 'rf_sigma_loss_w', 0.1)),
            sigma_log_loss=bool(getattr(config.model, 'rf_sigma_log_loss', True)),
            joint_eta_sigma=bool(getattr(config.model, 'rf_joint_eta_sigma', False)),
            joint_log_coeff=float(getattr(config.model, 'rf_joint_log_coeff', 1.0)),
            endpoint_mode=str(getattr(config.model, 'rf_endpoint_mode', 'zero')),
            target_space=str(getattr(config.model, 'rf_target_space', 'raw')),
            generation_scope=str(
                getattr(config.model, 'rf_generation_scope', 'future_only')
            ),
            eval_variant=str(
                getattr(config.model, 'rf_eval_variant', 'flow')
            ),
            ls_hidden=int(getattr(config.model, 'rf_ls_hidden', 128)),
            ls_graph_mode=str(getattr(config.model, 'rf_ls_graph_mode', 'graph')),
            ls_mu_scale=float(getattr(config.model, 'rf_ls_mu_scale', 0.1)),
            ls_sigma_min=float(getattr(config.model, 'rf_ls_sigma_min', 1e-3)),
            ls_sigma_max=float(getattr(config.model, 'rf_ls_sigma_max', 5.0)),
            ls_loss_weight=float(getattr(config.model, 'rf_ls_loss_w', 1.0)),
            ls_detach_target=bool(getattr(config.model, 'rf_ls_detach_target', True)),
            ls_warmup_epochs=int(getattr(config.model, 'rf_ls_warmup_epochs', 0)),
            ls_pretrain_epochs=int(getattr(config.model, 'rf_ls_pretrain_epochs', 5)),
            ls_freeze_after_pretrain=bool(getattr(config.model, 'rf_ls_freeze_after_pretrain', True)),
            delta=float(getattr(config.model, 'rf_delta', 1e-3)),
            uans_enable=bool(getattr(config.model, 'rf_uans', False)),
            uans_alpha=float(getattr(config.model, 'rf_uans_alpha', 0.0)),
            uans_min_pow=float(getattr(config.model, 'rf_uans_min_pow', 0.5)),
            uans_max_pow=float(getattr(config.model, 'rf_uans_max_pow', 2.0)),
            uans_floor=float(getattr(config.model, 'rf_uans_floor', 0.0)),
            weight_schedule=str(getattr(config.model, 'rf_w', 'none')),
            loss_type=str(getattr(config.model, 'rf_loss', 'x')),
            sample_chunk=int(getattr(config.model, 'rf_sample_chunk', 10)),
        )
        model = RFSDE_XPred(config.model, rf_cfg=rf_cfg)
    else:
        raise ValueError(f"Unknown engine: {config.engine}")
    model = model.to(config.device)
    model.training_protocol = checkpoint_protocol(config, clean_data)

    if bool(getattr(config, 'rf_gaussian_ls_only', False)):
        if config.engine != 'rf_sde_xpred':
            raise ValueError('Gaussian LS-only training requires rf_sde_xpred')
        if str(config.model.rf_target_space) != 'location_scale':
            raise ValueError("Gaussian LS-only training requires rf_target_space='location_scale'")
        if str(config.model.rf_eval_variant) != 'gaussian_ls':
            raise ValueError("Gaussian LS-only training requires rf_eval_variant='gaussian_ls'")
        if int(config.model.rf_ls_pretrain_epochs) < 1:
            raise ValueError('Gaussian LS-only training requires rf_ls_pretrain_epochs >= 1')
        if int(config.epoch) < int(config.model.rf_ls_pretrain_epochs):
            raise ValueError('epoch must cover all Gaussian LS pretraining epochs')
    elif (
        bool(params.get('is_train', True))
        and config.engine == 'rf_sde_xpred'
        and str(config.model.rf_target_space) != 'raw'
        and bool(config.model.rf_ls_freeze_after_pretrain)
    ):
        if int(config.model.rf_ls_pretrain_epochs) < 1:
            raise ValueError('Two-stage residual flow requires rf_ls_pretrain_epochs >= 1')
        if int(config.epoch) <= int(config.model.rf_ls_pretrain_epochs):
            raise ValueError('epoch must exceed rf_ls_pretrain_epochs so Stage 2 can train the residual flow')
        required_epochs = (
            int(config.model.rf_ls_pretrain_epochs)
            + int(config.rf_min_residual_epochs)
        )
        if int(config.epoch) < required_epochs:
            raise ValueError(
                'epoch must be at least rf_ls_pretrain_epochs + '
                'rf_min_residual_epochs for eligible checkpoint selection'
            )

    # Load training dataset
    train_dataset = TrafficDataset(clean_data, (config.model.T_h, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    val_dataset = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_h, config.data.test_start_idx - config.model.T_p + 1), config)
    # val_dataset   = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.val_start_idx + config.model.T_p + 512), config)
    # NOTE: For probabilistic engines (e.g., diffusion / RF-SDE) evaluation can be much more
    # memory-hungry than training due to multi-sample generation. Use the same batch size as
    # training to avoid CUDA OOM during validation/testing.
    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False, pin_memory=True)

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_h, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.batch_size, shuffle=False, pin_memory=True)


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                 if config.lr_scheduler == 'plateau' else None)

    # metrics in val, and test dataset, note that we cannot evaluate the performance in the train dataset
    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    model_path = config.PATH_MOD + config.trial_name + model.model_file_name()
    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)
    if model_path.endswith('.rfstg'):
        ls_pretrain_path = model_path[:-6] + '+stage-ls_pretrain.rfstg'
        best_mae_path = model_path[:-6] + '+best-mae.rfstg'
        last_model_path = model_path[:-6] + '+last.rfstg'
    else:
        ls_pretrain_path = model_path + '.ls_pretrain'
        best_mae_path = model_path + '.best_mae'
        last_model_path = model_path + '.last'
    config.ls_pretrain_path = ls_pretrain_path
    config.best_mae_path = best_mae_path
    config.last_model_path = last_model_path
    config.logger.write(f"ls pretrain path:{ls_pretrain_path}\n", is_terminal=False)
    config.logger.write(f"best MAE path:{best_mae_path}\n", is_terminal=False)
    config.logger.write(f"last model path:{last_model_path}\n", is_terminal=False)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{model_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)


    # log model architecture
    print(model)
    config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    config.logger.write(message, is_terminal=True)

    message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    train_start_t = timer()
    best_ls_objective = np.inf
    best_ls_epoch = -1
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params['is_train']: break
        if epoch > 1 and config.is_test: break

        if (
            config.engine == 'rf_sde_xpred'
            and str(config.model.rf_target_space) != 'raw'
            and epoch == int(config.model.rf_ls_pretrain_epochs)
            and best_ls_epoch >= 0
        ):
            pretrained_model = torch.load(ls_pretrain_path, map_location=config.device, weights_only=False)
            model.location_scale_head.load_state_dict(pretrained_model.location_scale_head.state_dict())
            del pretrained_model
            config.logger.write(
                f"restored best location-scale head from epoch {best_ls_epoch + 1} "
                f"(validation objective={best_ls_objective:.6f})\n",
                is_terminal=True,
            )

        train_stage = 'joint'
        if hasattr(model, 'set_train_epoch'):
            train_stage = model.set_train_epoch(epoch)
        model.train()

        n, avg_loss, time_lst = 0, 0, []
        # train model
        for i, batch in enumerate(train_loader):
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            x = torch.cat((history, future), dim=1).to(config.device) #  (B, T, V, F)

            # get x0_masked
            mask =  torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)# mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) # (B, T, V, F)

            # reshape
            x = x.transpose(1,3) # (B, F, V, T)
            x_masked = x_masked.transpose(1,3) # (B, F, V, T)

            # loss calculate
            if config.engine == 'diffstg':
                loss = 10 * model.loss(x, (x_masked, pos_w, pos_d))
            else:
                # RF-SDE engine already returns averaged loss
                loss = model.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            print('\r' + message, end='', flush=True)

        config.logger.message_buffer += message

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass

        if train_stage == 'location_scale_pretrain':
            ls_metrics = eval_location_scale_head(model, val_loader, config, clean_data)
            ls_metric_name = 'ValNLL'
            suffix = (
                f" | LS-pretrain {epoch + 1}/{int(config.model.rf_ls_pretrain_epochs)}"
                f" {ls_metric_name} {ls_metrics['objective']:.4f}"
                f" muMAE {ls_metrics['mu_mae']:.2f}"
                f" sigma {ls_metrics['sigma_mean']:.2f}"
                f" R({ls_metrics['residual_mean']:.3f},{ls_metrics['residual_std']:.3f})\n"
            )
            print(suffix, end='', flush=True)
            config.logger.message_buffer += suffix
            config.logger.write_message_buffer()
            if ls_metrics['objective'] < best_ls_objective:
                best_ls_objective = float(ls_metrics['objective'])
                best_ls_epoch = int(epoch)
                model.checkpoint_epoch = int(epoch)
                model.checkpoint_stage = 'location_scale_pretrain'
                model.ls_pretrain_best_epoch = int(best_ls_epoch)
                model.ls_pretrain_best_objective = float(best_ls_objective)
                model.checkpoint_selection_metric = str(ls_metric_name).lower()
                model.checkpoint_selection_value = float(best_ls_objective)
                torch.save(model, ls_pretrain_path)
                config.logger.write(
                    f"best location-scale checkpoint updated at epoch {epoch + 1}: {ls_pretrain_path}\n",
                    is_terminal=False,
                )
            if epoch + 1 == int(config.model.rf_ls_pretrain_epochs):
                config.logger.write(
                    f"location-scale pretraining complete; selected epoch {best_ls_epoch + 1} "
                    f"with validation objective={best_ls_objective:.6f}\n",
                    is_terminal=True,
                )
                if bool(getattr(config, 'rf_gaussian_ls_only', False)):
                    import shutil
                    shutil.copyfile(ls_pretrain_path, model_path)
                    config.logger.write(
                        f"Gaussian LS-only training complete; copied selected "
                        f"validation-NLL checkpoint to {model_path}\n",
                        is_terminal=True,
                    )
                    break
            continue

        residual_epochs = max(
            0,
            epoch - int(config.model.rf_ls_pretrain_epochs) + 1,
        )
        selection_eligible = not (
            config.engine == 'rf_sde_xpred'
            and str(config.model.rf_target_space) != 'raw'
            and residual_epochs < int(config.rf_min_residual_epochs)
        )

        if epoch >= config.start_epoch:
            evals(
                model,
                val_loader,
                epoch,
                metrics_val,
                config,
                clean_data,
                mode='Val',
                seed=config.seed,
                selection_eligible=selection_eligible,
            )
            if selection_eligible and scheduler is not None:
                scheduler.step(metrics_val.metrics[config.selection_metric])

        if metrics_val.last_improved.get(config.selection_metric, False):
            model.checkpoint_epoch = int(epoch)
            model.checkpoint_stage = str(train_stage)
            model.checkpoint_selection_metric = str(config.selection_metric)
            model.checkpoint_selection_value = float(metrics_val.metrics[config.selection_metric])
            torch.save(model, model_path)
            config.logger.write(
                f"best {config.selection_metric} checkpoint updated at epoch {epoch + 1} "
                f"(value={metrics_val.metrics[config.selection_metric]:.6f})\n",
                is_terminal=False,
            )

        if metrics_val.last_improved.get('mae', False):
            model.checkpoint_epoch = int(epoch)
            model.checkpoint_stage = str(train_stage)
            model.checkpoint_selection_metric = 'mae'
            model.checkpoint_selection_value = float(metrics_val.metrics['mae'])
            torch.save(model, best_mae_path)

        model.checkpoint_epoch = int(epoch)
        model.checkpoint_stage = str(train_stage)
        model.checkpoint_selection_metric = 'last'
        model.checkpoint_selection_value = float('nan')
        torch.save(model, last_model_path)

        best_epoch = metrics_val.best_metrics['epoch']
        if np.isfinite(best_epoch) and should_early_stop(epoch, int(best_epoch), config.early_stop):
            break  # Early_stop follows the configured selection metric.


    load_path = config.ckpt_path if (config.ckpt_path and (not bool(params.get('is_train', True)))) else model_path
    try:
        model = torch.load(load_path, map_location=config.device, weights_only=False)
        print('best model loaded from: <<', load_path)
        loaded_epoch = int(getattr(model, 'checkpoint_epoch', getattr(model, 'train_epoch', 0)))
        loaded_stage = str(getattr(model, 'checkpoint_stage', 'unknown'))
        loaded_selection_metric = str(getattr(model, 'checkpoint_selection_metric', 'unknown'))
        loaded_selection_value = getattr(model, 'checkpoint_selection_value', float('nan'))
        print(
            f'loaded checkpoint epoch: {loaded_epoch + 1}, stage: {loaded_stage}, '
            f'selection: {loaded_selection_metric}={loaded_selection_value}'
        )
        config.logger.write(
            f"loaded checkpoint epoch:{loaded_epoch + 1}, stage:{loaded_stage}, "
            f"selection:{loaded_selection_metric}={loaded_selection_value}\n",
            is_terminal=False,
        )
    except Exception as err:
        raise RuntimeError(f'Cannot load selected checkpoint: {load_path}') from err

    validate_checkpoint_protocol(model, checkpoint_protocol(config, clean_data))
    if config.model.paper_profile in ('www2027', 'canonical_ls_residual') and loaded_selection_metric != 'crps':
        raise ValueError('Paper evaluation requires the Stage-II checkpoint selected by validation CRPS')
    if hasattr(model, 'rf_cfg'):
        for field, expected in (
            ('target_space', config.model.rf_target_space),
            ('generation_scope', config.model.rf_generation_scope),
            ('ls_graph_mode', config.model.rf_ls_graph_mode),
            ('ls_detach_target', config.model.rf_ls_detach_target),
            ('ls_freeze_after_pretrain', config.model.rf_ls_freeze_after_pretrain),
            ('ls_pretrain_epochs', config.model.rf_ls_pretrain_epochs),
            ('standard_source', config.model.rf_standard_source),
            ('transport_reduction', config.model.rf_transport_reduction),
            ('sigma_mode', config.model.rf_sigma_mode),
            ('endpoint_mode', config.model.rf_endpoint_mode),
            ('mu_clip', config.model.rf_mu_clip),
            ('ls_sigma_min', config.model.rf_ls_sigma_min),
            ('ls_sigma_max', config.model.rf_ls_sigma_max),
            ('ls_mu_scale', config.model.rf_ls_mu_scale),
            ('ls_hidden', config.model.rf_ls_hidden),
            ('ls_loss_weight', config.model.rf_ls_loss_w),
            ('mu_mode', config.model.rf_mu_mode),
            ('sampling_mode', config.model.rf_mode),
            ('g_schedule', config.model.rf_g),
            ('noise_scale', config.model.rf_noise_scale),
            ('delta', config.model.rf_delta),
            ('uans_enable', config.model.rf_uans),
            ('loss_type', config.model.rf_loss),
            ('weight_schedule', config.model.rf_w),
        ):
            actual = getattr(model.rf_cfg, field, None)
            if actual != expected:
                raise ValueError(f'Checkpoint {field} mismatch: saved={actual!r}, requested={expected!r}')

    # Allow overriding inference-time sampling chunk size from CLI/config without retraining.
    # This is helpful for large graphs / high n_samples settings to avoid CUDA OOM.
    if hasattr(model, 'rf_cfg') and getattr(model, 'rf_cfg', None) is not None:
        loaded_generation_scope = (
            model._generation_scope()
            if hasattr(model, '_generation_scope')
            else 'future_only'
        )
        requested_generation_scope = str(
            getattr(config.model, 'rf_generation_scope', 'future_only')
        )
        if config.ckpt_path and requested_generation_scope != loaded_generation_scope:
            raise ValueError(
                'Checkpoint generation scope mismatch: requested '
                f'{requested_generation_scope}, checkpoint uses '
                f'{loaded_generation_scope}'
            )
        config.model.rf_generation_scope = loaded_generation_scope
        loaded_ls_graph_mode = (
            model._ls_graph_mode()
            if hasattr(model, '_ls_graph_mode')
            else 'graph'
        )
        requested_ls_graph_mode = str(
            getattr(config.model, 'rf_ls_graph_mode', 'graph')
        )
        if (
            config.ckpt_path
            and hasattr(model, '_target_space')
            and model._target_space() != 'raw'
            and requested_ls_graph_mode != loaded_ls_graph_mode
        ):
            raise ValueError(
                'Checkpoint LS graph mode mismatch: requested '
                f'{requested_ls_graph_mode}, checkpoint uses '
                f'{loaded_ls_graph_mode}'
            )
        config.model.rf_ls_graph_mode = loaded_ls_graph_mode
        model.rf_cfg.eval_variant = str(
            getattr(config.model, 'rf_eval_variant', 'flow')
        )
        model.rf_cfg.sample_chunk = int(getattr(config.model, 'rf_sample_chunk', getattr(model.rf_cfg, 'sample_chunk', 10)))
        # Keep inference-time behavior consistent with CLI flags even when loading a checkpoint.
        model.rf_cfg.sampling_mode = str(getattr(config.model, 'rf_mode', getattr(model.rf_cfg, 'sampling_mode', 'sde')))
        model.rf_cfg.g_schedule = 'square' if getattr(config.model, 'rf_g', 'linear') == 'square' else 'linear'
        model.rf_cfg.noise_scale = float(getattr(config.model, 'rf_noise_scale', getattr(model.rf_cfg, 'noise_scale', 1.0)))
        if getattr(config, 'rf_eval_sigma', None) is not None:
            model.rf_cfg.sigma = float(config.rf_eval_sigma)
        # Record the effective value rather than the parser default when a
        # checkpoint supplies sigma or a test-time override is requested.
        config.model.rf_sigma = float(getattr(model.rf_cfg, 'sigma', config.model.rf_sigma))
        model.rf_cfg.delta = float(getattr(config.model, 'rf_delta', getattr(model.rf_cfg, 'delta', 1e-3)))
        model.rf_cfg.uans_enable = bool(getattr(config.model, 'rf_uans', getattr(model.rf_cfg, 'uans_enable', False)))
        model.rf_cfg.uans_alpha = float(getattr(config.model, 'rf_uans_alpha', getattr(model.rf_cfg, 'uans_alpha', 0.0)))
        model.rf_cfg.uans_min_pow = float(getattr(config.model, 'rf_uans_min_pow', getattr(model.rf_cfg, 'uans_min_pow', 0.5)))
        model.rf_cfg.uans_max_pow = float(getattr(config.model, 'rf_uans_max_pow', getattr(model.rf_cfg, 'uans_max_pow', 2.0)))
        model.rf_cfg.uans_floor = float(getattr(config.model, 'rf_uans_floor', getattr(model.rf_cfg, 'uans_floor', 0.0)))
    # Ensure internal device attributes are consistent with map_location.
    if hasattr(model, 'device'):
        model.device = config.device

    calibration_mode = str(
        getattr(config, 'rf_residual_calibration', 'none')
    ).lower()
    if calibration_mode != 'none':
        if config.engine != 'rf_sde_xpred':
            raise ValueError('Residual spread calibration is only available for rf_sde_xpred')
        model_target_space = (
            model._target_space()
            if hasattr(model, '_target_space')
            else str(getattr(config.model, 'rf_target_space', 'raw'))
        )
        if model_target_space == 'raw':
            raise ValueError(
                'Residual spread calibration requires a mean_centered or '
                'location_scale checkpoint'
            )
        calibration_samples = int(getattr(config, 'rf_residual_calibration_samples', 0))
        preset_factors = list(
            getattr(config, 'rf_residual_calibration_factors', []) or []
        )
        if preset_factors:
            expected = 1 if calibration_mode == 'global' else int(config.model.T_p)
            if len(preset_factors) != expected:
                raise ValueError(
                    f'{calibration_mode} calibration expects {expected} pre-fitted '
                    f'factor(s), got {len(preset_factors)}'
                )
            if not all(np.isfinite(value) and value > 0 for value in preset_factors):
                raise ValueError('Pre-fitted residual calibration factors must be finite and positive')
            calibration_factors = [float(value) for value in preset_factors]
            print(f'using pre-fitted {calibration_mode} residual spread calibration ...')
        else:
            if calibration_samples <= 0:
                calibration_samples = int(config.n_samples)
            print(
                f'fitting {calibration_mode} residual spread calibration on validation '
                f'with S={calibration_samples} ...'
            )
            calibration_factors = fit_residual_spread_calibration(
                model,
                val_loader,
                config,
                clean_data,
                seed=int(config.seed) + 7919,
            )
            config.rf_residual_calibration_factors = calibration_factors
        calibration_message = (
            f'residual spread calibration mode:{calibration_mode}, '
            f'samples:{calibration_samples if calibration_samples > 0 else "prefit"}, factors:'
            + ','.join(f'{value:.6f}' for value in calibration_factors)
            + '\n'
        )
        print(calibration_message, end='')
        config.logger.write(calibration_message, is_terminal=False)

    # conduct multiple-samples, then report the best
    metric_lst = []
    test_results = []
    if config.engine == 'diffstg':
        # Respect the requested inference-time sampler configuration.  The
        # previous hard-coded DDIM-40 setting silently ignored
        # --sample_steps and made controlled NFE comparisons impossible.
        for sample_strategy, sample_steps in [(
            str(config.model.sample_strategy),
            int(config.model.sample_steps),
        )]:
            if sample_steps > config.model.N: break

            config.model.sample_strategy = sample_strategy
            config.model.sample_steps = sample_steps

            model.set_ddim_sample_steps(sample_steps)
            model.set_sample_strategy(sample_strategy)

            metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
            evals(model, test_loader, loaded_epoch, metrics_test, config, clean_data, mode='test', seed=config.seed)
            message = f'sample_strategy:{sample_strategy}, sample_steps:{sample_steps} Final results in test:{metrics_test}\n'
            config.logger.write(message, is_terminal=True)

            params = unfold_dict(config)
            params = dict_merge([params, metrics_test.to_dict()])
            params['best_epoch'] = metrics_val.best_metrics['epoch']
            params['checkpoint_selection_metric'] = loaded_selection_metric
            params['checkpoint_selection_value'] = float(loaded_selection_value)
            params['model'] = config.model.epsilon_theta
            save2file(params)
            metric_lst.append(metrics_test.metrics['mae'])
            result_record = {
                'paper_profile': config.model.paper_profile,
                'paper_variant': config.paper_variant,
                'paper_experiment': config.paper_experiment,
                'crps_definition': 'empirical_cdf_integral',
                'eval_clip_nonneg': config.eval_clip_nonneg,
                'dataset': config.data.name,
                'engine': config.engine,
                'seed': int(config.seed),
                'eval_tag': config.eval_tag,
                'eval_window_limit': int(getattr(config, 'eval_window_limit', 0)),
                'checkpoint': load_path,
                'sampling_steps': int(sample_steps),
                'effective_nfe': int(sample_steps),
                'n_samples': int(config.n_samples),
                'batch_size': int(config.batch_size),
                'num_parameters_total': int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                'num_parameters_effective': int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                'nominal_coverage': float(1.0 - config.interval_alpha),
                'best_epoch': int(metrics_val.best_metrics['epoch']) if np.isfinite(metrics_val.best_metrics['epoch']) else None,
                **{k: float(v) for k, v in metrics_test.metrics.items()},
            }
            if bool(getattr(config, 'detailed_eval', False)):
                result_record['diagnostics'] = metrics_test.diagnostics
            test_results.append(result_record)
    else:
        # RF-SDE engine: optionally sweep sampling steps for efficiency analysis.
        rf_steps_grid = list(config.rf_eval_steps_grid) if getattr(config, 'rf_eval_steps_grid', None) else []
        if not rf_steps_grid:
            rf_steps_grid = [int(config.model.rf_steps)]

        original_steps = int(config.model.rf_steps)
        original_model_steps = int(getattr(model.rf_cfg, 'num_sampling_steps', original_steps))
        for rf_steps in rf_steps_grid:
            config.model.rf_steps = int(rf_steps)
            if hasattr(model, 'rf_cfg'):
                model.rf_cfg.num_sampling_steps = int(rf_steps)

            metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
            evals(model, test_loader, loaded_epoch, metrics_test, config, clean_data, mode='test', seed=config.seed)
            message = (
                f'engine:{config.engine}, rf_mode:{config.model.rf_mode}, '
                f'rf_steps:{rf_steps}, rf_target_space:{config.model.rf_target_space}, '
                f'rf_generation_scope:{config.model.rf_generation_scope} '
                f'rf_eval_variant:{config.model.rf_eval_variant} '
                f'rf_ls_graph_mode:{config.model.rf_ls_graph_mode} '
                f'checkpoint_epoch:{loaded_epoch + 1}, checkpoint_stage:{loaded_stage} '
                f'Final results in test:{metrics_test}\n'
            )
            config.logger.write(message, is_terminal=True)

            params = unfold_dict(config)
            params = dict_merge([params, metrics_test.to_dict()])
            params['best_epoch'] = metrics_val.best_metrics['epoch']
            params['checkpoint_selection_metric'] = loaded_selection_metric
            params['checkpoint_selection_value'] = float(loaded_selection_value)
            params['model'] = config.engine
            save2file(params)
            metric_lst.append(metrics_test.metrics['mae'])
            result_record = {
                'paper_profile': config.model.paper_profile,
                'paper_variant': config.paper_variant,
                'paper_experiment': config.paper_experiment,
                'crps_definition': 'empirical_cdf_integral',
                'eval_clip_nonneg': config.eval_clip_nonneg,
                'dataset': config.data.name,
                'engine': config.engine,
                'seed': int(config.seed),
                'eval_tag': config.eval_tag,
                'checkpoint': load_path,
                'checkpoint_epoch': int(loaded_epoch + 1),
                'checkpoint_stage': loaded_stage,
                'checkpoint_selection_metric': loaded_selection_metric,
                'checkpoint_selection_value': float(loaded_selection_value),
                'rf_steps': int(rf_steps),
                'n_samples': int(config.n_samples),
                'batch_size': int(config.batch_size),
                'nominal_coverage': float(1.0 - config.interval_alpha),
                'rf_sigma_mode': str(config.model.rf_sigma_mode),
                'rf_sigma': float(getattr(model.rf_cfg, 'sigma', config.model.rf_sigma)),
                'rf_target_space': str(config.model.rf_target_space),
                'rf_generation_scope': str(
                    config.model.rf_generation_scope
                ),
                'rf_eval_variant': str(config.model.rf_eval_variant),
                'rf_ls_graph_mode': str(config.model.rf_ls_graph_mode),
                'effective_nfe': int(
                    0 if config.model.rf_eval_variant == 'gaussian_ls' else rf_steps
                ),
                'num_parameters_total': int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                'num_parameters_effective': int(
                    sum(
                        parameter.numel()
                        for parameter in (
                            model.location_scale_head.parameters()
                            if config.model.rf_eval_variant == 'gaussian_ls'
                            else model.parameters()
                        )
                    )
                ),
                'rf_ls_hidden': int(config.model.rf_ls_hidden),
                'rf_ls_mu_scale': float(config.model.rf_ls_mu_scale),
                'rf_ls_sigma_min': float(config.model.rf_ls_sigma_min),
                'rf_ls_sigma_max': float(config.model.rf_ls_sigma_max),
                'rf_ls_loss_w': float(config.model.rf_ls_loss_w),
                'rf_ls_detach_target': bool(config.model.rf_ls_detach_target),
                'rf_ls_warmup_epochs': int(config.model.rf_ls_warmup_epochs),
                'rf_ls_pretrain_epochs': int(config.model.rf_ls_pretrain_epochs),
                'rf_ls_freeze_after_pretrain': bool(config.model.rf_ls_freeze_after_pretrain),
                'val_n_samples': int(config.val_n_samples),
                'selection_metric': str(config.selection_metric),
                'rf_min_residual_epochs': int(config.rf_min_residual_epochs),
                'rf_uans': bool(config.model.rf_uans),
                'rf_uans_alpha': float(config.model.rf_uans_alpha),
                'rf_residual_calibration': str(config.rf_residual_calibration),
                'rf_residual_calibration_samples': int(
                    config.rf_residual_calibration_samples
                    if config.rf_residual_calibration_samples > 0
                    else config.n_samples
                ),
                'rf_residual_calibration_factors': [
                    float(value)
                    for value in config.rf_residual_calibration_factors
                ],
                'best_epoch': int(metrics_val.best_metrics['epoch']) if np.isfinite(metrics_val.best_metrics['epoch']) else None,
                **{k: float(v) for k, v in metrics_test.metrics.items()},
            }
            if bool(getattr(config, 'detailed_eval', False)):
                result_record['diagnostics'] = metrics_test.diagnostics
            test_results.append(result_record)

        config.model.rf_steps = original_steps
        if hasattr(model, 'rf_cfg'):
            model.rf_cfg.num_sampling_steps = original_model_steps

    if config.result_json:
        result_path = os.path.abspath(os.path.expanduser(config.result_json))
        dir_check(result_path)
        payload = {
            'protocol': {
                'paper_profile': config.model.paper_profile,
                'paper_variant': config.paper_variant,
                'paper_experiment': config.paper_experiment,
                'crps_definition': 'empirical_cdf_integral',
                'eval_clip_nonneg': config.eval_clip_nonneg,
                'dataset': config.data.name,
                'split': {
                    'tag': str(config.data.split_tag),
                    'train': float(config.data.train_ratio),
                    'validation': float(config.data.val_ratio),
                    'test': float(config.data.test_ratio),
                    'val_start_idx': int(config.data.val_start_idx),
                    'test_start_idx': int(config.data.test_start_idx),
                },
                'seed': int(config.seed),
                'is_train': bool(config.is_train),
                'checkpoint': load_path,
                'n_samples': int(config.n_samples),
                'batch_size': int(config.batch_size),
                'val_n_samples': int(config.val_n_samples),
                'selection_metric': str(config.selection_metric),
                'rf_min_residual_epochs': int(config.rf_min_residual_epochs),
                'nominal_coverage': float(1.0 - config.interval_alpha),
                'weight_decay': float(config.wd),
                'lr_scheduler': config.lr_scheduler,
                'training_protocol': model.training_protocol,
                'timing_scope': 'input_transfer_condition_generation_output_transfer; excludes denormalization and metrics',
                'eval_warmup_batches': config.eval_warmup_batches,
                'device': str(config.device),
                'gpu_name': torch.cuda.get_device_name(config.device) if config.device.type == 'cuda' else None,
                'rf_standard_source': config.model.rf_standard_source,
                'transport_reduction': config.model.rf_transport_reduction,
                'eval_tag': config.eval_tag,
                'rf_residual_calibration': str(config.rf_residual_calibration),
                'rf_residual_calibration_samples': int(
                    config.rf_residual_calibration_samples
                    if config.rf_residual_calibration_samples > 0
                    else config.n_samples
                ),
                'rf_residual_calibration_factors': [
                    float(value)
                    for value in config.rf_residual_calibration_factors
                ],
                'rf_generation_scope': str(
                    config.model.rf_generation_scope
                ),
                'rf_eval_variant': str(config.model.rf_eval_variant),
                'rf_ls_graph_mode': str(config.model.rf_ls_graph_mode),
                'rf_gaussian_ls_only': bool(config.rf_gaussian_ls_only),
                'detailed_eval': bool(getattr(config, 'detailed_eval', False)),
                'detailed_coverage_levels': [
                    float(value) for value in getattr(
                        config, 'detailed_coverage_levels', []
                    )
                ],
            },
            'results': test_results,
        }
        with open(result_path, 'w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):7.2f}+{log_name}")
    import shutil
    # os.rename(config.log_path, new_log_path)
    shutil.copy(config.log_path, new_log_path)
    config.log_path = new_log_path

    try:
        writer.close()
    except:
        pass

    if config.nni and (nni is not None):
        nni.report_final_result(min(metric_lst))


# data.name	model	model.N	model.epsilon_theta	model.d_h	model.T_h	model.T_p	model.sample_strategy
# PEMS08	UGnet	200	    UGnet	            32	        12	        12	        ddpm

if __name__ == '__main__':
    import logging

    logger = logging.getLogger('training')

    print('CUDA available:', torch.cuda.is_available())
    try:
        params = vars(get_params())
        # Merge NNI tuner params only when NNI is available.
        if nni is not None:
            try:
                tuner_params = nni.get_next_parameter()
                logger.debug(tuner_params)
                if tuner_params:
                    params.update(tuner_params)
            except Exception as e:
                logger.warning(f"NNI parameter fetch failed, fallback to CLI params only: {e}")
        elif params.get('nni', False):
            logger.warning("NNI is not available in current environment. Running without NNI integration.")
            params['nni'] = False
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
