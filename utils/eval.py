# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from timeit import default_timer as timer


# Targets are evaluated after float32 normalization followed by inverse
# normalization.  An original zero can therefore reappear as a tiny residual
# (for example 1.5258789e-5 on PEMS08) instead of an exact 0.  Treat values in
# this numerical neighbourhood as the null value so they never enter a MAPE
# denominator.  The smallest positive PEMS08 flow is 1.0, well above this
# tolerance.
ZERO_NULL_ATOL = 1e-4


def _valid_nonzero_target_mask_torch(y_true, atol=ZERO_NULL_ATOL):
    """Return finite targets that are materially different from zero."""
    return torch.isfinite(y_true) & (y_true.abs() > float(atol))


def resolve_eval_n_samples(config, mode):
    """Return the configured ensemble size for validation or final testing."""
    if str(mode).lower() == 'val':
        value = getattr(config, 'val_n_samples', 8)
    else:
        value = getattr(config, 'n_samples', 100)
    value = int(value)
    if value < 1:
        raise ValueError('Evaluation sample count must be at least 1')
    return value


def apply_residual_spread_calibration(y_pred, factors):
    """Scale ensemble deviations without changing their sample mean.

    Parameters
    ----------
    y_pred:
        Tensor shaped ``(B, S, T, V, D)``.
    factors:
        A scalar or one positive factor per prediction horizon.  Horizon-wise
        factors are broadcast over batch, samples, nodes, and features.

    Notes
    -----
    This transformation is deliberately performed in observation space after
    reverse normalization.  Consequently the factors have a direct
    calibration interpretation and are invariant to affine dataset scaling.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
    if y_pred.ndim != 5:
        raise ValueError(
            'Residual spread calibration expects predictions shaped '
            f'(B,S,T,V,D), got {tuple(y_pred.shape)}'
        )

    scale = torch.as_tensor(factors, dtype=y_pred.dtype, device=y_pred.device).flatten()
    if scale.numel() == 1:
        scale = scale.expand(y_pred.shape[2])
    if scale.numel() != y_pred.shape[2]:
        raise ValueError(
            f'Expected one calibration factor or T={y_pred.shape[2]} factors, '
            f'got {scale.numel()}'
        )
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Residual spread calibration factors must be finite and positive')

    center = y_pred.mean(dim=1, keepdim=True)
    return center + scale.view(1, 1, -1, 1, 1) * (y_pred - center)


class CoverageSpreadCalibrator:
    """Fit global or horizon-wise ensemble spread factors on validation data.

    For every validation target, the required multiplicative expansion is
    measured relative to the central ensemble interval.  The conformal
    empirical quantile of these scores becomes the calibration factor.  No
    test target is used, and scaling around the ensemble mean preserves point
    forecasts before an optional non-negativity clamp.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        mode: str = 'horizon',
        min_factor: float = 0.5,
        max_factor: float = 4.0,
        eps: float = 1e-6,
    ):
        self.alpha = float(alpha)
        self.mode = str(mode).lower()
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)
        self.eps = float(eps)
        if not 0.0 < self.alpha < 1.0:
            raise ValueError('alpha must be in (0, 1)')
        if self.mode not in ('global', 'horizon'):
            raise ValueError("calibration mode must be 'global' or 'horizon'")
        if self.min_factor <= 0 or self.max_factor < self.min_factor:
            raise ValueError('Invalid residual spread calibration bounds')
        self._scores = None

    def update(self, y_true, y_pred):
        """Accumulate required spread multipliers from one validation batch."""
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        y_true = y_true.detach().to(dtype=torch.float32, device='cpu')
        y_pred = y_pred.detach().to(dtype=torch.float32, device='cpu')
        if y_pred.ndim != 5 or y_true.ndim != 4:
            raise ValueError('Expected y_true=(B,T,V,D), y_pred=(B,S,T,V,D)')
        if y_pred.shape[0] != y_true.shape[0] or y_pred.shape[2:] != y_true.shape[1:]:
            raise ValueError(
                f'Calibration shape mismatch: true={tuple(y_true.shape)}, '
                f'pred={tuple(y_pred.shape)}'
            )

        center = y_pred.mean(dim=1)
        lower = torch.quantile(y_pred, self.alpha / 2.0, dim=1)
        upper = torch.quantile(y_pred, 1.0 - self.alpha / 2.0, dim=1)
        lower_spread = (center - lower).clamp_min(self.eps)
        upper_spread = (upper - center).clamp_min(self.eps)
        required = torch.where(
            y_true >= center,
            (y_true - center).clamp_min(0.0) / upper_spread,
            (center - y_true).clamp_min(0.0) / lower_spread,
        )
        required = torch.nan_to_num(
            required,
            nan=self.max_factor,
            posinf=self.max_factor,
            neginf=self.min_factor,
        )

        if self.mode == 'global':
            batch_scores = [required.reshape(-1)]
        else:
            batch_scores = [required[:, horizon].reshape(-1) for horizon in range(required.shape[1])]
        if self._scores is None:
            self._scores = [[] for _ in batch_scores]
        if len(self._scores) != len(batch_scores):
            raise ValueError('Prediction horizon changed while fitting calibration')
        for storage, scores in zip(self._scores, batch_scores):
            storage.append(scores)

    def compute_factors(self):
        """Return split-conformal finite-sample quantiles of required spread."""
        if not self._scores or any(not chunks for chunks in self._scores):
            raise RuntimeError('No validation predictions were provided to the calibrator')
        factors = []
        for chunks in self._scores:
            scores = torch.cat(chunks).flatten()
            scores = scores[torch.isfinite(scores)]
            if scores.numel() == 0:
                raise RuntimeError('No finite calibration scores were observed')
            # Split-conformal order statistic: ceil((n+1)*(1-alpha)).
            rank = int(np.ceil((scores.numel() + 1) * (1.0 - self.alpha)))
            rank = min(max(rank, 1), int(scores.numel()))
            factor = float(torch.kthvalue(scores, rank).values.item())
            factor = min(max(factor, self.min_factor), self.max_factor)
            factors.append(factor)
        return factors


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    if float(null_val) == 0.0:
        array = np.asarray(array)
        return (
            np.isfinite(array) & (np.abs(array) > ZERO_NULL_ATOL)
        ).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask_mean = mask.mean()
        if mask_mean < 1e-8:
            return 0.0
        mask /= mask_mean
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask_mean = mask.mean()
    if mask_mean < 1e-8:
        return 0.0
    mask /= mask_mean
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask_mean = mask.mean()
    if mask_mean < 1e-8:
        return 0.0
    mask /= mask_mean
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Metric(object):
    """Computes and stores the average and current value,调用时先reset"""

    def __init__(self, T_p):
        self.time_start = timer()
        self.T_p = T_p
        self.selection_metric = 'mae'
        self.last_improved = {}
        self.best_metrics = {
            'mae': np.inf,
            'rmse': np.inf,
            'mape': np.inf,
            'crps': np.inf,
            'mis': np.inf,
            'picp': np.inf,
            'sharpness': np.inf,
            'interval_width': np.inf,
            'calib_error': np.inf,
            'epoch': np.inf,
            'selection_value': np.inf,
        }
        self.metrics = {}
        # self.step_metrics_epoch = {'mae': {}, 'rmse': {}, 'mape': {}}


    def update_metrics(self, y_true, y_pred):
        accumulator = StreamingMetricAccumulator(alpha=0.1)
        accumulator.update(y_true, y_pred)
        self.metrics = accumulator.compute_metrics()
        self.metrics['time'] = timer() - self.time_start

    def update_best_metrics(self, epoch=0, selection_metric='mae', eligible=True):
        """Update per-metric minima and the epoch used for checkpoint selection.

        ``eligible=False`` is used during the residual-flow warm-up. Those
        validation values are still logged, but they cannot become a best
        checkpoint or influence early stopping.
        """
        selection_metric = str(selection_metric).lower()
        tracked = (
            'mae', 'rmse', 'mape', 'crps', 'mis',
            'sharpness', 'interval_width', 'calib_error',
        )
        if selection_metric not in tracked:
            raise ValueError(
                f"Unsupported selection_metric={selection_metric!r}; "
                f"expected one of {tracked}"
            )

        self.selection_metric = selection_metric
        self.last_improved = {name: False for name in tracked}
        if not eligible:
            return False

        for name in tracked:
            best, improved = self.get_best_metric(
                self.best_metrics[name], self.metrics[name]
            )
            self.best_metrics[name] = best
            self.last_improved[name] = improved

        if self.last_improved[selection_metric]:
            self.best_metrics['epoch'] = int(epoch)
            self.best_metrics['selection_value'] = float(self.metrics[selection_metric])
            self.best_metrics['picp'] = self.metrics['picp']
            return True
        return False

    @staticmethod
    def get_metric(y_true, y_pred):
        mae = masked_mae_np(y_true, y_pred, np.nan)
        mse = masked_mse_np(y_true, y_pred, np.nan)
        mape = masked_mape_np(y_true, y_pred, 0)
        rmse = mse ** 0.5
        return mae, rmse, mape, mse

    @staticmethod
    def get_best_metric(best, candidate):
        state = False
        if candidate < best:

            best = candidate
            state = True
        return best, state

    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['crps']:<7.2f} {self.metrics['mis']:<7.2f} | {self.best_metrics['epoch'] + 1:<4} "

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f}"


    def log_lst(self, epoch=None, sep=','):
        message_lst = []
        index = ['mae', 'rmse', 'mape']

        for i in index:
            message_lst.append(f"{i},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")
        return message_lst

    def to_dict(self):
        return self.metrics


def empirical_crps_values(target, forecast):
    """Exact integral of the empirical CDF (TeX Eq. 22), per target entry.

    E|X-y| - 0.5 E|X-X'|, with two independent draws from the empirical
    distribution. Sorted samples avoid materializing an S by S tensor.
    """
    if forecast.ndim == target.ndim:
        forecast = forecast.unsqueeze(1)
    if forecast.shape[0] != target.shape[0] or forecast.shape[2:] != target.shape[1:]:
        raise ValueError('Forecast/target shape mismatch')
    count = forecast.shape[1]
    if count < 1:
        raise ValueError('At least one forecast sample is required')
    ordered = forecast.sort(dim=1).values
    weights = 2 * torch.arange(1, count + 1, device=forecast.device, dtype=forecast.dtype) - count - 1
    weights = weights.reshape(1, count, *([1] * (forecast.ndim - 2)))
    pair_term = (ordered * weights).sum(dim=1) / (count * count)
    return (forecast - target.unsqueeze(1)).abs().mean(dim=1) - pair_term


class StreamingMetricAccumulator:
    """Physical-unit empirical CRPS and point/interval metrics over entries.

    Missing targets are excluded consistently. Observed zeros enter MAE,
    RMSE, CRPS and intervals; only MAPE omits near-zero denominators.
    """

    def __init__(self, alpha=0.1, detailed=False,
                 coverage_levels=(0.5, 0.8, 0.9, 0.95), store_case_metrics=False):
        self.alpha = float(alpha)
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must be in (0,1)')
        self.detailed = bool(detailed)
        self.coverage_levels = sorted(set(round(float(x), 8) for x in (*coverage_levels, 1-self.alpha)))
        if any(not 0 < level < 1 for level in self.coverage_levels):
            raise ValueError('coverage levels must be in (0,1)')
        self.store_case_metrics = bool(store_case_metrics)
        self.reset()

    def reset(self):
        self.sums = dict.fromkeys(('count', 'mape_count', 'mae', 'mse', 'mape',
                                  'crps', 'mis', 'picp', 'width', 'sharpness'), 0.0)
        self.horizon_sums = None
        self.coverage_sums = None
        self.case_components = []

    def update(self, y_true, y_pred):
        y_true = torch.as_tensor(y_true).detach().to(dtype=torch.float64, device='cpu')
        y_pred = torch.as_tensor(y_pred).detach().to(dtype=torch.float64, device='cpu')
        if y_pred.ndim == y_true.ndim:
            y_pred = y_pred.unsqueeze(1)
        if not torch.isfinite(y_pred).all():
            raise ValueError('Forecast contains non-finite values')
        valid = torch.isfinite(y_true)
        safe_target = torch.where(valid, y_true, torch.zeros_like(y_true))
        percentage_valid = valid & (safe_target.abs() > ZERO_NULL_ATOL)
        err = y_pred.mean(dim=1) - safe_target
        lower = torch.quantile(y_pred, self.alpha/2, dim=1)
        upper = torch.quantile(y_pred, 1-self.alpha/2, dim=1)
        width = upper-lower
        score = width + 2/self.alpha * (lower-safe_target).clamp_min(0) + 2/self.alpha * (safe_target-upper).clamp_min(0)
        values = {
            'count': valid.double(), 'mape_count': percentage_valid.double(),
            'mae': err.abs(), 'mse': err.square(),
            'mape': torch.where(percentage_valid, err.abs()/safe_target.abs().clamp_min(1e-12), 0.0),
            'crps': empirical_crps_values(safe_target, y_pred),
            'mis': score, 'picp': ((safe_target >= lower) & (safe_target <= upper)).double(),
            'width': width, 'sharpness': y_pred.std(dim=1, unbiased=False),
        }
        values = {key: torch.where(valid, value, 0.0) for key, value in values.items()}
        for key, value in values.items():
            self.sums[key] += value.sum().item()
        if not self.detailed:
            return
        if self.horizon_sums is None:
            self.horizon_sums = {key: np.zeros(y_true.shape[1], dtype=np.float64) for key in values}
            self.coverage_sums = {str(level): {key: np.zeros(y_true.shape[1]) for key in ('count','mis','picp','width')} for level in self.coverage_levels}
        if len(self.horizon_sums['count']) != y_true.shape[1]:
            raise ValueError('Prediction horizon changed during detailed evaluation')
        for key, value in values.items():
            self.horizon_sums[key] += value.sum(dim=(0,2,3)).numpy()
        for level in self.coverage_levels:
            alpha = 1-level
            lo = torch.quantile(y_pred, alpha/2, dim=1)
            hi = torch.quantile(y_pred, 1-alpha/2, dim=1)
            level_values = {
                'count': valid.double(), 'width': hi-lo,
                'mis': hi-lo+2/alpha*(lo-safe_target).clamp_min(0)+2/alpha*(safe_target-hi).clamp_min(0),
                'picp': ((safe_target >= lo) & (safe_target <= hi)).double(),
            }
            for key, value in level_values.items():
                self.coverage_sums[str(level)][key] += torch.where(valid, value, 0.0).sum(dim=(0,2,3)).numpy()
        if self.store_case_metrics:
            names = {'count':'valid_count', 'mape_count':'mape_count',
                     'mae':'mae_sum', 'mse':'mse_sum', 'mape':'mape_sum',
                     'crps':'crps_sum', 'mis':'mis_sum', 'picp':'picp_sum',
                     'width':'width_sum', 'sharpness':'sharpness_sum'}
            component = {names[key]: value.sum(dim=(1,2,3)).numpy() for key,value in values.items()}
            component['interval_count'] = component['valid_count'].copy()
            self.case_components.append(component)

    def _metrics(self, sums):
        count = max(float(sums['count']), 1.0)
        picp = float(sums['picp']/count)
        return {'mae': float(sums['mae']/count), 'rmse': float((sums['mse']/count)**0.5),
                'mape': float(sums['mape']/max(float(sums['mape_count']),1.0)*100),
                'crps': float(sums['crps']/count), 'mis': float(sums['mis']/count),
                'picp': picp, 'sharpness': float(sums['sharpness']/count),
                'interval_width': float(sums['width']/count),
                'calib_error': abs(picp-(1-self.alpha)), 'time': 0.0}

    def compute_metrics(self):
        if self.sums['count'] == 0:
            raise ValueError('No observed targets were evaluated')
        metrics = self._metrics(self.sums)
        if abs(self.alpha - 0.1) < 1e-12:
            metrics.update(picp90=metrics['picp'], piw90=metrics['interval_width'], mis90=metrics['mis'])
        return metrics

    def compute_diagnostics(self):
        if self.horizon_sums is None:
            return {}
        horizons = []
        for i in range(len(self.horizon_sums['count'])):
            metrics = self._metrics({key:value[i] for key,value in self.horizon_sums.items()})
            metrics.pop('time')
            horizons.append({'horizon':i+1, **metrics})
        profile, horizon_coverage = [], {}
        for level in self.coverage_levels:
            values = self.coverage_sums[str(level)]
            count = max(values['count'].sum(),1.0)
            picp = values['picp'].sum()/count
            profile.append({'nominal_coverage': level, 'picp': float(picp),
                            'calib_error': float(abs(picp-level)),
                            'mis': float(values['mis'].sum()/count),
                            'interval_width': float(values['width'].sum()/count)})
            horizon_coverage[str(level)] = [
                {'horizon':i+1, 'picp':float(values['picp'][i]/max(c,1)),
                 'mis':float(values['mis'][i]/max(c,1)),
                 'interval_width':float(values['width'][i]/max(c,1))}
                for i,c in enumerate(values['count'])]
        out = {'horizon_metrics':horizons, 'coverage_profile':profile, 'horizon_coverage':horizon_coverage}
        if self.case_components:
            out['case_components'] = {key:np.concatenate([c[key] for c in self.case_components]).tolist() for key in self.case_components[0]}
        return out


class ConditionalResidualDiagnosticAccumulator:
    """Stream point-mu and raw-vs-residual distribution diagnostics."""

    def __init__(self):
        self.count_grid = None
        self.target_sum_grid = None
        self.target_sq_sum_grid = None
        self.residual_sum_grid = None
        self.residual_sq_sum_grid = None
        self.point_mae_sum_grid = None
        self.point_mse_sum_grid = None
        self.point_mape_sum_grid = None
        self.mape_count_grid = None

    def _initialize(self, horizons, nodes):
        shape = (int(horizons), int(nodes))
        self.count_grid = np.zeros(shape, dtype=np.float64)
        self.target_sum_grid = np.zeros(shape, dtype=np.float64)
        self.target_sq_sum_grid = np.zeros(shape, dtype=np.float64)
        self.residual_sum_grid = np.zeros(shape, dtype=np.float64)
        self.residual_sq_sum_grid = np.zeros(shape, dtype=np.float64)
        self.point_mae_sum_grid = np.zeros(shape, dtype=np.float64)
        self.point_mse_sum_grid = np.zeros(shape, dtype=np.float64)
        self.point_mape_sum_grid = np.zeros(shape, dtype=np.float64)
        self.mape_count_grid = np.zeros(shape, dtype=np.float64)

    def update(self, y_true, mu_pred):
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(mu_pred, np.ndarray):
            mu_pred = torch.from_numpy(mu_pred)
        y_true = y_true.detach().to(dtype=torch.float32, device='cpu')
        mu_pred = mu_pred.detach().to(dtype=torch.float32, device='cpu')
        if y_true.shape != mu_pred.shape or y_true.ndim != 4:
            raise ValueError('Expected matching y_true/mu_pred shaped (B,T,V,D)')
        if self.count_grid is None:
            self._initialize(y_true.shape[1], y_true.shape[2])

        valid = torch.isfinite(y_true)
        mape_valid = _valid_nonzero_target_mask_torch(y_true)
        y_true = torch.where(valid, y_true, torch.zeros_like(y_true))
        valid_float = valid.to(dtype=torch.float32)
        residual = y_true - mu_pred
        reduce_dims = (0, 3)
        self.count_grid += valid_float.sum(dim=reduce_dims).numpy()
        self.mape_count_grid += mape_valid.to(dtype=torch.float32).sum(dim=reduce_dims).numpy()
        self.target_sum_grid += (y_true * valid_float).sum(dim=reduce_dims).numpy()
        self.target_sq_sum_grid += (y_true.square() * valid_float).sum(dim=reduce_dims).numpy()
        self.residual_sum_grid += (residual * valid_float).sum(dim=reduce_dims).numpy()
        self.residual_sq_sum_grid += (residual.square() * valid_float).sum(dim=reduce_dims).numpy()
        self.point_mae_sum_grid += (residual.abs() * valid_float).sum(dim=reduce_dims).numpy()
        self.point_mse_sum_grid += (residual.square() * valid_float).sum(dim=reduce_dims).numpy()
        point_mape = torch.where(
            mape_valid,
            residual.abs() / y_true.abs().clamp_min(1e-12),
            torch.zeros_like(residual),
        )
        self.point_mape_sum_grid += point_mape.sum(dim=reduce_dims).numpy()

    @staticmethod
    def _moments(value_sum, square_sum, count):
        safe_count = np.maximum(count, 1.0)
        mean = value_sum / safe_count
        variance = np.maximum(square_sum / safe_count - mean ** 2, 0.0)
        return mean, np.sqrt(variance)

    def compute(self):
        if self.count_grid is None:
            return {}
        target_mean, target_std = self._moments(
            self.target_sum_grid, self.target_sq_sum_grid, self.count_grid
        )
        residual_mean, residual_std = self._moments(
            self.residual_sum_grid, self.residual_sq_sum_grid, self.count_grid
        )
        total_count = max(float(self.count_grid.sum()), 1.0)
        point_mae = float(self.point_mae_sum_grid.sum() / total_count)
        point_rmse = float((self.point_mse_sum_grid.sum() / total_count) ** 0.5)
        point_mape = float(self.point_mape_sum_grid.sum() / max(float(self.mape_count_grid.sum()), 1.0) * 100.0)

        horizon_count = np.maximum(self.count_grid.sum(axis=1), 1.0)
        horizon_target_mean, horizon_target_std = self._moments(
            self.target_sum_grid.sum(axis=1),
            self.target_sq_sum_grid.sum(axis=1),
            horizon_count,
        )
        horizon_residual_mean, horizon_residual_std = self._moments(
            self.residual_sum_grid.sum(axis=1),
            self.residual_sq_sum_grid.sum(axis=1),
            horizon_count,
        )
        horizon_point_mae = self.point_mae_sum_grid.sum(axis=1) / horizon_count
        horizon_point_rmse = np.sqrt(
            self.point_mse_sum_grid.sum(axis=1) / horizon_count
        )

        node_count = np.maximum(self.count_grid.sum(axis=0), 1.0)
        node_target_mean, node_target_std = self._moments(
            self.target_sum_grid.sum(axis=0),
            self.target_sq_sum_grid.sum(axis=0),
            node_count,
        )
        node_residual_mean, node_residual_std = self._moments(
            self.residual_sum_grid.sum(axis=0),
            self.residual_sq_sum_grid.sum(axis=0),
            node_count,
        )

        return {
            'point_mu': {
                'mae': point_mae,
                'rmse': point_rmse,
                'mape': point_mape,
                'horizon_mae': horizon_point_mae.tolist(),
                'horizon_rmse': horizon_point_rmse.tolist(),
            },
            'horizon': {
                'target_mean': horizon_target_mean.tolist(),
                'target_std': horizon_target_std.tolist(),
                'residual_mean': horizon_residual_mean.tolist(),
                'residual_std': horizon_residual_std.tolist(),
            },
            'node': {
                'target_mean': node_target_mean.tolist(),
                'target_std': node_target_std.tolist(),
                'residual_mean': node_residual_mean.tolist(),
                'residual_std': node_residual_std.tolist(),
            },
            'grid': {
                'target_mean': target_mean.tolist(),
                'target_std': target_std.tolist(),
                'residual_mean': residual_mean.tolist(),
                'residual_std': residual_std.tolist(),
            },
            'summary': {
                'target_abs_grid_mean': float(np.mean(np.abs(target_mean))),
                'residual_abs_grid_mean': float(np.mean(np.abs(residual_mean))),
                'target_mean_grid_std': float(np.mean(target_std)),
                'residual_mean_grid_std': float(np.mean(residual_std)),
                'target_grid_std_cv': float(np.std(target_std) / max(np.mean(target_std), 1e-12)),
                'residual_grid_std_cv': float(np.std(residual_std) / max(np.mean(residual_std), 1e-12)),
            },
        }


# metric for Probabilistic evaluation
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_empirical_CRPS(target, forecast, eval_points=None):
    values = empirical_crps_values(target, forecast)
    valid = torch.isfinite(target)
    if eval_points is not None:
        valid = valid & (eval_points > 0)
    if not valid.any():
        raise ValueError('No observed targets for CRPS')
    return float(values[valid].mean().item())


def calc_quantile_CRPS(target, forecast, eval_points):
    """Compatibility name: returns ordinary empirical CRPS, without normalization."""
    return calc_empirical_CRPS(target, forecast, eval_points)


def MIS(
        target: np.ndarray,
        lower_quantile: np.ndarray,
        upper_quantile: np.ndarray,
        alpha: float,


) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator


def calc_mis(target, forecast, alpha = 0.1):
    """
       target: (B, T, V),
       forecast: (B, n_sample, T, V)
    """
    return MIS(
        target = target.cpu().numpy(),
        lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
        upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
        alpha = alpha,
    )


def calc_interval_metrics(target, forecast, alpha=0.1):
    """
    target: (B, T, V, D) or (B, T, V)
    forecast: (B, n_sample, T, V, D) or (B, n_sample, T, V)
    Returns interval-based probabilistic metrics computed on the central (1-alpha) interval.
    """
    lower = torch.quantile(forecast, alpha / 2, dim=1)
    upper = torch.quantile(forecast, 1.0 - alpha / 2, dim=1)
    nominal = 1.0 - alpha

    inside = ((target >= lower) & (target <= upper)).float()
    picp = float(inside.mean().item())
    interval_width = float((upper - lower).mean().item())
    sharpness = float(forecast.std(dim=1, unbiased=False).mean().item())
    calib_error = abs(picp - nominal)

    return {
        'picp': picp,
        'sharpness': sharpness,
        'interval_width': interval_width,
        'calib_error': calib_error,
    }


if __name__ == "__main__":
    # test for CRPS
    B, T, V = 32, 12, 36
    n_sample = 100
    target = torch.randn((B, T, V))
    forecast = torch.randn((B, n_sample, T, V))
    label = target.unsqueeze(1).expand_as(forecast)
    eval_points = torch.randn_like(target)

    crps = calc_quantile_CRPS(target, forecast, eval_points)
    print('crps:', crps)

    crps = calc_quantile_CRPS(target, label, eval_points)
    print('crps:', crps)

    mis = calc_mis(target, forecast)
    print('mis:', mis)

    mis = calc_mis(target, label)
    print('mis:', mis)
