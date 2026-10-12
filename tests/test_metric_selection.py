import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.eval import (
    ConditionalResidualDiagnosticAccumulator,
    CoverageSpreadCalibrator,
    Metric,
    StreamingMetricAccumulator,
    apply_residual_spread_calibration,
    masked_mape_np,
    resolve_eval_n_samples,
)


def set_metrics(metric, *, mae, crps):
    metric.metrics = {
        'mae': float(mae),
        'rmse': float(mae) + 1.0,
        'mape': float(mae) + 2.0,
        'crps': float(crps),
        'mis': 10.0,
        'picp': 0.8,
        'sharpness': 2.0,
        'interval_width': 5.0,
        'calib_error': 0.15,
        'time': 0.0,
    }


class MetricSelectionTest(unittest.TestCase):
    def test_validation_and_test_use_distinct_ensemble_sizes(self):
        config = SimpleNamespace(val_n_samples=8, n_samples=100)
        self.assertEqual(resolve_eval_n_samples(config, 'Val'), 8)
        self.assertEqual(resolve_eval_n_samples(config, 'val'), 8)
        self.assertEqual(resolve_eval_n_samples(config, 'test'), 100)

    def test_ineligible_warmup_does_not_select_checkpoint(self):
        metric = Metric(T_p=12)
        set_metrics(metric, mae=1.0, crps=0.1)

        improved = metric.update_best_metrics(
            epoch=3,
            selection_metric='crps',
            eligible=False,
        )

        self.assertFalse(improved)
        self.assertTrue(np.isinf(metric.best_metrics['epoch']))
        self.assertTrue(np.isinf(metric.best_metrics['crps']))
        self.assertFalse(any(metric.last_improved.values()))

    def test_crps_selects_main_epoch_while_mae_tracks_own_minimum(self):
        metric = Metric(T_p=12)

        set_metrics(metric, mae=2.0, crps=0.20)
        self.assertTrue(metric.update_best_metrics(4, 'crps', eligible=True))
        self.assertTrue(metric.last_improved['mae'])
        self.assertEqual(metric.best_metrics['epoch'], 4)

        set_metrics(metric, mae=1.5, crps=0.25)
        self.assertFalse(metric.update_best_metrics(5, 'crps', eligible=True))
        self.assertTrue(metric.last_improved['mae'])
        self.assertFalse(metric.last_improved['crps'])
        self.assertEqual(metric.best_metrics['epoch'], 4)
        self.assertEqual(metric.best_metrics['mae'], 1.5)

        set_metrics(metric, mae=1.8, crps=0.15)
        self.assertTrue(metric.update_best_metrics(6, 'crps', eligible=True))
        self.assertFalse(metric.last_improved['mae'])
        self.assertTrue(metric.last_improved['crps'])
        self.assertEqual(metric.best_metrics['epoch'], 6)
        self.assertEqual(metric.best_metrics['selection_value'], 0.15)

    def test_unsupported_selection_metric_is_rejected(self):
        metric = Metric(T_p=12)
        set_metrics(metric, mae=1.0, crps=0.1)
        with self.assertRaisesRegex(ValueError, 'Unsupported selection_metric'):
            metric.update_best_metrics(0, 'picp', eligible=True)

    def test_spread_calibration_preserves_ensemble_mean(self):
        predictions = torch.tensor(
            [[[[[1.0]], [[2.0]]], [[[3.0]], [[6.0]]]]],
            dtype=torch.float32,
        )
        calibrated = apply_residual_spread_calibration(predictions, [2.0, 0.5])
        self.assertTrue(
            torch.allclose(
                predictions.mean(dim=1),
                calibrated.mean(dim=1),
                atol=1e-6,
            )
        )
        original_deviation = predictions - predictions.mean(dim=1, keepdim=True)
        calibrated_deviation = calibrated - calibrated.mean(dim=1, keepdim=True)
        self.assertTrue(
            torch.allclose(
                calibrated_deviation[:, :, 0],
                2.0 * original_deviation[:, :, 0],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                calibrated_deviation[:, :, 1],
                0.5 * original_deviation[:, :, 1],
                atol=1e-6,
            )
        )

    def test_horizon_calibrator_recovers_distinct_underdispersion(self):
        # With two symmetric samples, the central 90% interval is [-0.9, 0.9].
        # The two targets therefore require factors 2 and 1 respectively.
        predictions = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        predictions = predictions.view(1, 2, 1, 1, 1).repeat(20, 1, 2, 1, 1)
        targets = torch.tensor([1.8, 0.9], dtype=torch.float32)
        targets = targets.view(1, 2, 1, 1).repeat(20, 1, 1, 1)
        calibrator = CoverageSpreadCalibrator(
            alpha=0.1,
            mode='horizon',
            min_factor=0.5,
            max_factor=4.0,
        )
        calibrator.update(targets, predictions)
        factors = calibrator.compute_factors()
        self.assertAlmostEqual(factors[0], 2.0, places=5)
        self.assertAlmostEqual(factors[1], 1.0, places=5)

    def test_detailed_streaming_metrics_include_horizons_and_cases(self):
        target = torch.tensor(
            [
                [[[1.0]], [[2.0]]],
                [[[3.0]], [[4.0]]],
            ]
        )
        predictions = torch.stack([target - 0.5, target + 0.5], dim=1)
        accumulator = StreamingMetricAccumulator(
            alpha=0.1,
            detailed=True,
            coverage_levels=[0.5, 0.9],
            store_case_metrics=True,
        )
        accumulator.update(target, predictions)
        metrics = accumulator.compute_metrics()
        diagnostics = accumulator.compute_diagnostics()

        self.assertAlmostEqual(metrics['mae'], 0.0, places=6)
        self.assertEqual(len(diagnostics['horizon_metrics']), 2)
        self.assertEqual(len(diagnostics['coverage_profile']), 2)
        self.assertEqual(
            len(diagnostics['case_components']['valid_count']), 2
        )

    def test_inverse_normalization_roundoff_is_excluded_from_mape(self):
        # PEMS08 raw zeros can become this value after float32 inverse scaling.
        roundoff_zero = 1.52587890625e-5
        target = np.array([roundoff_zero, 1.0], dtype=np.float32)
        prediction = np.array([100.0, 2.0], dtype=np.float32)

        self.assertAlmostEqual(
            masked_mape_np(target, prediction, null_val=0),
            100.0,
            places=6,
        )

        streaming_target = torch.tensor(
            [[[[roundoff_zero]], [[1.0]]]], dtype=torch.float32
        )
        streaming_prediction = torch.tensor(
            [[[[[100.0]], [[2.0]]]]], dtype=torch.float32
        )
        accumulator = StreamingMetricAccumulator(detailed=True)
        accumulator.update(streaming_target, streaming_prediction)
        metrics = accumulator.compute_metrics()
        diagnostics = accumulator.compute_diagnostics()

        self.assertAlmostEqual(metrics['mae'], (100.0-roundoff_zero+1.0)/2, places=6)
        self.assertAlmostEqual(metrics['rmse'], (((100.0-roundoff_zero)**2+1.0)/2)**0.5, places=6)
        self.assertAlmostEqual(metrics['mape'], 100.0, places=6)
        self.assertEqual(diagnostics['horizon_metrics'][0]['mape'], 0.0)
        self.assertAlmostEqual(
            diagnostics['horizon_metrics'][1]['mape'], 100.0, places=6
        )

    def test_conditional_residual_diagnostic_reports_point_mu(self):
        target = torch.tensor(
            [
                [[[2.0]], [[4.0]]],
                [[[4.0]], [[8.0]]],
            ]
        )
        mu = target - 1.0
        accumulator = ConditionalResidualDiagnosticAccumulator()
        accumulator.update(target, mu)
        diagnostics = accumulator.compute()

        self.assertAlmostEqual(diagnostics['point_mu']['mae'], 1.0, places=6)
        self.assertAlmostEqual(diagnostics['point_mu']['rmse'], 1.0, places=6)
        self.assertEqual(len(diagnostics['horizon']['residual_mean']), 2)

    def test_conditional_residual_mape_excludes_roundoff_zero(self):
        roundoff_zero = 1.52587890625e-5
        target = torch.tensor(
            [[[[roundoff_zero]], [[1.0]]]], dtype=torch.float32
        )
        mu = torch.tensor(
            [[[[100.0]], [[0.0]]]], dtype=torch.float32
        )
        accumulator = ConditionalResidualDiagnosticAccumulator()
        accumulator.update(target, mu)
        diagnostics = accumulator.compute()

        self.assertAlmostEqual(diagnostics['point_mu']['mae'], (100.0-roundoff_zero+1.0)/2, places=5)
        self.assertAlmostEqual(diagnostics['point_mu']['rmse'], (((100.0-roundoff_zero)**2+1.0)/2)**0.5, places=5)
        self.assertAlmostEqual(diagnostics['point_mu']['mape'], 100.0, places=6)


if __name__ == '__main__':
    unittest.main()
