import unittest

import numpy as np
import torch
from easydict import EasyDict

from algorithm.rf_sde.model import RFSDEConfig, RFSDEXPred


def make_stg_config():
    return EasyDict(
        T_h=4,
        T_p=4,
        V=3,
        F=1,
        d_h=8,
        supports_len=2,
        channel_multipliers=[1],
        n_blocks=1,
        A=np.eye(3, dtype=np.float32),
        device=torch.device("cpu"),
    )


def make_batch(batch_size=2):
    history = torch.randn(batch_size, 1, 3, 4)
    future = torch.randn(batch_size, 1, 3, 4)
    x_full = torch.cat([history, future], dim=-1)
    x_masked = torch.cat([history, torch.zeros_like(future)], dim=-1)
    pos_w = torch.zeros(batch_size, 4, dtype=torch.long)
    pos_d = torch.zeros(batch_size, 4, dtype=torch.long)
    return x_full, (x_masked, pos_w, pos_d)


class LocationScaleResidualFlowTest(unittest.TestCase):
    def test_location_scale_loss_backward_and_roundtrip(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                num_sampling_steps=2,
                ls_hidden=16,
            ),
        )
        x_full, cond = make_batch()
        loss = model.loss(x_full, cond)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.location_scale_head.scale_out.bias.grad)
        self.assertTrue(torch.isfinite(model.location_scale_head.scale_out.bias.grad).all())

        history, future = model._split_history_future(x_full)
        mu, sigma = model._compute_location_scale(history)
        residual = model._encode_future_target(future, mu, sigma, detach=True)
        reconstructed = model._decode_future_target(residual, mu, sigma)
        self.assertTrue(torch.allclose(reconstructed, future, atol=1e-5, rtol=1e-5))
        self.assertGreaterEqual(float(sigma.min()), model.rf_cfg.ls_sigma_min)
        self.assertLessEqual(float(sigma.max()), model.rf_cfg.ls_sigma_max)

    def test_location_scale_sampling_shape(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                num_sampling_steps=2,
                sample_chunk=1,
                ls_hidden=16,
            ),
        )
        _x_full, cond = make_batch(batch_size=1)
        samples = model.sample_future(cond, n_samples=2)
        self.assertEqual(tuple(samples.shape), (1, 2, 1, 3, 8))
        self.assertTrue(torch.isfinite(samples).all())

    def test_stop_gradient_blocks_flow_signal_to_ls_head(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                ls_hidden=16,
                ls_loss_weight=0.0,
                ls_detach_target=True,
            ),
        )
        x_full, cond = make_batch()
        model.loss(x_full, cond).backward()
        for parameter in model.location_scale_head.parameters():
            self.assertTrue(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0)

    def test_mean_centered_mode(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="mean_centered",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                num_sampling_steps=2,
                ls_hidden=16,
            ),
        )
        x_full, cond = make_batch()
        loss = model.loss(x_full, cond)
        self.assertTrue(torch.isfinite(loss))

    def test_graph_free_location_head_is_nodewise_with_equal_capacity(self):
        stg_config = make_stg_config()
        stg_config.A = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        common = dict(
            target_space="mean_centered",
            sigma_mode="const",
            endpoint_mode="zero",
            uans_enable=False,
            num_sampling_steps=2,
            ls_hidden=16,
        )
        graph_model = RFSDEXPred(
            stg_config,
            RFSDEConfig(**common, ls_graph_mode="graph"),
        )
        graph_free_model = RFSDEXPred(
            stg_config,
            RFSDEConfig(**common, ls_graph_mode="none"),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in graph_model.parameters()),
            sum(parameter.numel() for parameter in graph_free_model.parameters()),
        )
        identity = torch.eye(stg_config.V)
        self.assertTrue(
            torch.equal(graph_free_model.location_scale_head.adj_forward, identity)
        )
        self.assertTrue(
            torch.equal(graph_free_model.location_scale_head.adj_backward, identity)
        )

        history = torch.randn(2, 1, stg_config.V, stg_config.T_h)
        changed_neighbors = history.clone()
        changed_neighbors[:, :, 1:, :] += 100.0
        mu, sigma = graph_free_model._compute_location_scale(history)
        changed_mu, changed_sigma = graph_free_model._compute_location_scale(
            changed_neighbors
        )
        self.assertTrue(torch.equal(mu[:, :, 0], changed_mu[:, :, 0]))
        self.assertTrue(torch.equal(sigma[:, :, 0], changed_sigma[:, :, 0]))
        self.assertIn("lsgraph-none", graph_free_model.model_file_name())

        (mu.sum() + sigma.sum()).backward()
        self.assertIsNotNone(
            graph_free_model.location_scale_head.forward_proj.weight.grad
        )
        self.assertIsNotNone(
            graph_free_model.location_scale_head.backward_proj.weight.grad
        )

    def test_invalid_location_head_graph_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown ls_graph_mode"):
            RFSDEXPred(
                make_stg_config(),
                RFSDEConfig(
                    target_space="mean_centered",
                    sigma_mode="const",
                    endpoint_mode="zero",
                    ls_graph_mode="invalid",
                ),
            )

    def test_location_scale_pretrain_only_updates_ls_head(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                ls_hidden=16,
                ls_pretrain_epochs=2,
            ),
        )
        model.set_train_epoch(0)
        x_full, cond = make_batch()
        loss = model.loss(x_full, cond)
        loss.backward()
        self.assertIsNone(next(model.backbone.parameters()).grad)
        self.assertIsNotNone(model.location_scale_head.scale_out.bias.grad)

    def test_flow_stage_freezes_location_scale_coordinates(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                sigma_mode="const",
                endpoint_mode="zero",
                uans_enable=False,
                ls_hidden=16,
                ls_pretrain_epochs=2,
                ls_freeze_after_pretrain=True,
            ),
        )
        self.assertEqual(model.set_train_epoch(2), "residual_flow")
        self.assertFalse(any(p.requires_grad for p in model.location_scale_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
        x_full, cond = make_batch()
        model.loss(x_full, cond).backward()
        self.assertIsNotNone(next(model.backbone.parameters()).grad)
        for parameter in model.location_scale_head.parameters():
            self.assertIsNone(parameter.grad)

    def test_raw_mode_remains_available(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(target_space="raw", num_sampling_steps=2, sample_chunk=1),
        )
        x_full, cond = make_batch(batch_size=1)
        self.assertTrue(torch.isfinite(model.loss(x_full, cond)))
        samples = model.sample_future(cond, n_samples=1)
        self.assertEqual(tuple(samples.shape), (1, 1, 1, 3, 8))
        expected_history = cond[0][..., :4].cpu()
        self.assertTrue(torch.equal(samples[:, 0, ..., :4], expected_history))

    def test_joint_history_future_scope_uses_same_parameterization(self):
        future_only = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="raw",
                generation_scope="future_only",
                sigma_mode="const",
                endpoint_mode="zero",
                num_sampling_steps=2,
                sample_chunk=1,
            ),
        )
        joint = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="raw",
                generation_scope="joint_history_future",
                sigma_mode="const",
                endpoint_mode="zero",
                num_sampling_steps=2,
                sample_chunk=1,
            ),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in future_only.parameters()),
            sum(parameter.numel() for parameter in joint.parameters()),
        )

        x_full, cond = make_batch(batch_size=1)
        loss = joint.loss(x_full, cond)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(next(joint.backbone.parameters()).grad)

        samples = joint.sample_future(cond, n_samples=2)
        self.assertEqual(tuple(samples.shape), (1, 2, 1, 3, 8))
        self.assertTrue(torch.isfinite(samples).all())
        conditioned_history = cond[0][..., :4].cpu()
        self.assertFalse(torch.equal(samples[:, 0, ..., :4], conditioned_history))
        self.assertIn("scope-joint_history_future", joint.model_file_name())

    def test_joint_scope_rejects_residual_target_space(self):
        with self.assertRaisesRegex(ValueError, "requires target_space='raw'"):
            RFSDEXPred(
                make_stg_config(),
                RFSDEConfig(
                    target_space="mean_centered",
                    generation_scope="joint_history_future",
                    sigma_mode="const",
                    endpoint_mode="zero",
                ),
            )

    def test_gaussian_ls_sampling_uses_only_location_scale_head(self):
        model = RFSDEXPred(
            make_stg_config(),
            RFSDEConfig(
                target_space="location_scale",
                eval_variant="gaussian_ls",
                sigma_mode="const",
                endpoint_mode="zero",
                sample_chunk=1,
                ls_hidden=16,
            ),
        )

        class FailIfCalled(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                raise AssertionError("Flow backbone must not run for Gaussian LS")

        model.backbone = FailIfCalled()
        _x_full, cond = make_batch(batch_size=1)
        samples = model.sample_future(cond, n_samples=2)
        self.assertEqual(tuple(samples.shape), (1, 2, 1, 3, 8))
        self.assertTrue(torch.isfinite(samples).all())
        expected_history = cond[0][..., :4].cpu()
        self.assertTrue(torch.equal(samples[:, 0, ..., :4], expected_history))
        self.assertIn("eval-gaussian_ls", model.model_file_name())

    def test_gaussian_ls_rejects_raw_target_space(self):
        with self.assertRaisesRegex(ValueError, "requires target_space='location_scale'"):
            RFSDEXPred(
                make_stg_config(),
                RFSDEConfig(target_space="raw", eval_variant="gaussian_ls"),
            )

    def test_residual_mode_rejects_legacy_uans(self):
        with self.assertRaisesRegex(ValueError, "do not use UANS"):
            RFSDEXPred(
                make_stg_config(),
                RFSDEConfig(
                    target_space="location_scale",
                    sigma_mode="const",
                    endpoint_mode="zero",
                    uans_enable=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
