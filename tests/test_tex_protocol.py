import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict

import train
from algorithm.dataset import CleanDataset
from train import validate_checkpoint_protocol, should_early_stop
from algorithm.rf_sde.model import RFSDEConfig, RFSDEXPred
from utils.eval import StreamingMetricAccumulator, empirical_crps_values, calc_empirical_CRPS
from test_rf_sde_location_scale import make_stg_config, make_batch


def params_for(*args):
    with patch.object(sys, 'argv', ['train.py', *args]):
        return train.apply_paper_profile(vars(train.get_params()))


class ConstantEndpoint(torch.nn.Module):
    def __init__(self, value=2.0):
        super().__init__()
        self.value = value
        self.inputs = []

    def forward(self, state, time, condition):
        self.inputs.append(state.detach().clone())
        return torch.full_like(state, self.value)


class TexProtocolTest(unittest.TestCase):
    def test_defaults_and_dataset_gamma(self):
        for data in ('AIR_BJ', 'AIR_GZ', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'):
            cfg = params_for('--data', data)
            self.assertEqual(cfg['rf_sigma'], 0.08 if data.startswith('AIR') else 0.05)
            self.assertEqual((cfg['lr'], cfg['batch_size'], cfg['epoch']), (0.002, 8, 300))
            self.assertEqual((cfg['n_samples'],cfg['val_n_samples']), (100,8))
            self.assertEqual(cfg['rf_target_space'], 'location_scale')
            self.assertEqual(cfg['rf_ls_pretrain_epochs'], 5)
            self.assertEqual(cfg['rf_min_residual_epochs'], 1)
            self.assertEqual(cfg['rf_mu_clip'], 0)
            self.assertEqual(cfg['wd'], 0)
            self.assertEqual(cfg['lr_scheduler'], 'none')
            self.assertEqual(cfg['interval_alpha'], 0.1)
            self.assertFalse(cfg['eval_clip_nonneg'])
            self.assertEqual(cfg['rf_residual_calibration'], 'none')

    def test_all_ablations_keep_source_budget_and_backbone(self):
        for variant in ('ablation_full','joint_window_raw','future_only_raw','mean_centered','no_graph','no_freezing'):
            cfg = params_for('--paper_variant',variant)
            self.assertEqual(cfg['n_samples'],8)
            self.assertEqual((cfg['hidden_size'],cfg['rf_steps'],cfg['epoch']), (32,20,300))
            self.assertTrue(cfg['rf_standard_source'])
        joint=params_for('--paper_variant','no_freezing')
        self.assertEqual(joint['rf_ls_pretrain_epochs'],0)
        self.assertFalse(joint['rf_ls_freeze_after_pretrain'])
        self.assertFalse(joint['rf_ls_detach_target'])
        self.assertEqual(params_for('--paper_variant','no_graph')['rf_ls_graph_mode'],'none')
        self.assertEqual(params_for('--paper_variant','joint_window_raw')['rf_generation_scope'],'joint_history_future')

    def test_baseline_engine_keeps_its_own_entry(self):
        cfg=params_for('--engine','diffstg')
        self.assertEqual(cfg['engine'],'diffstg')
        self.assertEqual(cfg['paper_profile'],'none')

    def test_standard_protocol_cannot_be_changed_by_legacy_flags(self):
        cfg=params_for('--data','AIR_GZ','--lr','0.0005','--batch_size','32',
                       '--rf_target_space','raw','--rf_sigma','0.5',
                       '--rf_steps','5','--n_samples','8','--rf_ls_pretrain_epochs','1',
                       '--eval_clip_nonneg','true','--rf_residual_calibration','horizon')
        self.assertEqual((cfg['lr'],cfg['batch_size'],cfg['rf_sigma']),(.002,8,.08))
        self.assertEqual((cfg['rf_steps'],cfg['n_samples'],cfg['rf_ls_pretrain_epochs']),(20,100,5))
        self.assertEqual(cfg['rf_target_space'],'location_scale')
        self.assertFalse(cfg['eval_clip_nonneg'])
        self.assertEqual(cfg['rf_residual_calibration'],'none')

    def test_sensitivity_overrides_require_an_explicit_experiment(self):
        with self.assertRaisesRegex(ValueError,'sensitivity'):
            params_for('--rf_eval_sigma','0.15')
        cfg=params_for('--paper_experiment','sensitivity','--rf_steps','30',
                       '--rf_eval_sigma','0.15','--n_samples','200',
                       '--rf_ls_pretrain_epochs','10')
        self.assertEqual((cfg['rf_steps'],cfg['rf_eval_sigma'],cfg['n_samples'],cfg['rf_ls_pretrain_epochs']),
                         (30,.15,200,10))

    def test_experiment_commands_follow_the_protocol(self):
        from tools.run import build_commands
        base=dict(data='AIR_GZ', device='cpu', seed=2022,
                  output_dir='/tmp/faststg-command-test', checkpoint='/tmp/main.rfstg')
        ablations=build_commands(SimpleNamespace(task='ablation',**base))
        self.assertEqual(len(ablations),6)
        for command in ablations:
            cfg=params_for(*command[2:])
            self.assertEqual(cfg['n_samples'],8)
            self.assertEqual(cfg['interval_alpha'],0.1)
            self.assertFalse(cfg['eval_clip_nonneg'])
        commands=build_commands(SimpleNamespace(task='sensitivity',**base))
        self.assertEqual(len(commands),19)
        for command in commands:
            cfg=params_for(*command[2:])
            self.assertEqual(cfg['rf_target_space'],'location_scale')
            self.assertEqual(cfg['rf_sigma'],0.08)
        self.assertTrue(any('--n_samples' in c and c[c.index('--n_samples')+1]=='200' for c in commands))

    def test_empirical_crps_matches_pairwise_identity_and_scaling(self):
        torch.manual_seed(7)
        target=torch.randn(2,3,2,1,dtype=torch.float64)
        samples=torch.randn(2,7,3,2,1,dtype=torch.float64)
        brute=(samples-target[:,None]).abs().mean(1)-0.5*(samples[:,:,None]-samples[:,None,:]).abs().mean((1,2))
        self.assertTrue(torch.allclose(empirical_crps_values(target,samples),brute,atol=1e-12))
        self.assertAlmostEqual(calc_empirical_CRPS(10*target,10*samples),10*brute.mean().item(),places=10)
        self.assertEqual(calc_empirical_CRPS(torch.zeros(1,1),torch.tensor([[[0.],[2.]]])),0.5)

    def test_metrics_include_zero_and_mask_only_missing_targets(self):
        target=torch.tensor([0.,2.,float('nan')]).reshape(1,3,1,1)
        samples=torch.tensor([10.,2.,999.]).reshape(1,1,3,1,1)
        acc=StreamingMetricAccumulator(detailed=True,store_case_metrics=True)
        acc.update(target,samples)
        metrics=acc.compute_metrics()
        self.assertEqual(metrics['mae'],5)
        self.assertAlmostEqual(metrics['rmse'],(50)**0.5)
        self.assertEqual(metrics['crps'],5)
        self.assertEqual(metrics['mape'],0)
        self.assertEqual(acc.compute_diagnostics()['case_components']['valid_count'],[2])

    def test_streaming_and_diagnostics_are_partition_invariant(self):
        torch.manual_seed(1)
        target=torch.randn(5,4,2,1)
        samples=torch.randn(5,8,4,2,1)
        full=StreamingMetricAccumulator(detailed=True,store_case_metrics=True)
        split=StreamingMetricAccumulator(detailed=True,store_case_metrics=True)
        full.update(target,samples)
        for sl in (slice(0,2),slice(2,5)):
            split.update(target[sl],samples[sl])
        for key,value in full.compute_metrics().items():
            self.assertAlmostEqual(value,split.compute_metrics()[key],places=10)
        case=split.compute_diagnostics()['case_components']
        self.assertAlmostEqual(sum(case['crps_sum'])/sum(case['valid_count']),full.compute_metrics()['crps'],places=10)

    def test_interval90_formula(self):
        target=torch.tensor([[[[2.]]]])
        forecast=torch.tensor([-1.,1.]).reshape(1,2,1,1,1)
        acc=StreamingMetricAccumulator()
        acc.update(target,forecast)
        metrics=acc.compute_metrics()
        self.assertAlmostEqual(metrics['interval_width'],1.8)
        self.assertAlmostEqual(metrics['mis'],1.8+20*(2-.9))
        self.assertEqual(metrics['picp'],0)

    def test_mu_matches_unclipped_equation6(self):
        model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space='location_scale'))
        mu,_=model._compute_location_scale(torch.full((1,1,3,4),6.0))
        self.assertTrue(torch.equal(mu,torch.full_like(mu,6.0)))

    def test_transport_is_squared_l2_not_element_mean(self):
        model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space='location_scale'))
        self.assertEqual(model._transport_loss(torch.ones(2,1,3,4)).item(),12.0)

    def test_raw_source_does_not_change_with_gamma(self):
        _x,cond=make_batch(1)
        states=[]
        for gamma in (.02,.15):
            model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space='raw',sigma=gamma,num_sampling_steps=1))
            endpoint=ConstantEndpoint()
            model.backbone=endpoint
            torch.manual_seed(19)
            model.sample_future(cond,1)
            states.append(endpoint.inputs[0][...,-4:])
        self.assertTrue(torch.equal(*states))
        self.assertGreater(float(states[0].std()),0.5)

    def test_training_bridges_use_the_same_standard_source(self):
        x,cond=make_batch(1)
        for space,scope in (('raw','future_only'),('raw','joint_history_future'),('location_scale','future_only')):
            model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space=space,generation_scope=scope,sigma=.08))
            model.backbone=ConstantEndpoint()
            if space=='location_scale':
                model.set_train_epoch(5)
                history,future=model._split_history_future(x)
                mu,sigma=model._compute_location_scale(history)
                target=model._encode_future_target(future,mu,sigma,True)
            else:
                target=x if scope=='joint_history_future' else x[...,-4:]
            with patch('torch.rand',side_effect=lambda shape,**kw:torch.full(shape,.25,device=kw.get('device'))), \
                 patch('torch.randn_like',side_effect=lambda value:torch.ones_like(value)):
                model.loss(x,cond)
            state=model.backbone.inputs[0]
            if scope=='future_only':
                state=state[...,-4:]
            self.assertTrue(torch.allclose(state,.25*target+.75,atol=1e-6))

    def test_em_updates_and_inverse_transform_match_algorithm2(self):
        model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space='location_scale',sigma=.08,num_sampling_steps=2,sample_chunk=1))
        model.backbone=ConstantEndpoint(2.0)
        _x,cond=make_batch(1)
        calls=[]
        def coordinates(history):
            calls.append(1)
            shape=(*history.shape[:-1],4)
            return torch.full(shape,3.0),torch.full(shape,2.0)
        with patch.object(model,'_compute_location_scale',side_effect=coordinates), \
             patch('torch.randn',side_effect=lambda shape,**kw:torch.ones(shape,device=kw.get('device'))), \
             patch('torch.randn_like',side_effect=lambda x:torch.full_like(x,2.0)):
            samples=model.sample_future(cond,3)
        z=1.0
        for k in range(2):
            t=k*.5
            z+=.5*(2-z)/(1-t+.001)+.08*(1-t)*(.5**.5)*2
        self.assertEqual(len(calls),1)
        self.assertTrue(torch.allclose(samples[...,-4:],torch.full_like(samples[...,-4:],3+2*z),atol=1e-6))
        self.assertTrue(torch.equal(samples[:,0,...,:4],cond[0][...,:4]))

    def test_joint_ablation_allows_transport_gradients_to_head(self):
        model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space='location_scale',ls_pretrain_epochs=0,ls_freeze_after_pretrain=False,ls_detach_target=False,ls_loss_weight=0))
        self.assertEqual(model.set_train_epoch(0),'joint')
        x,cond=make_batch()
        model.loss(x,cond).backward()
        self.assertGreater(float(model.location_scale_head.scale_out.bias.grad.abs().sum()),0)

    def test_missing_targets_do_not_create_nonfinite_loss_or_gradients(self):
        for space in ('raw','mean_centered','location_scale'):
            model=RFSDEXPred(make_stg_config(),RFSDEConfig(target_space=space))
            x,cond=make_batch()
            x[0,0,0,-1]=float('nan')
            loss=model.loss(x,cond)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_missing_fill_is_causal_and_node_mean_uses_train(self):
        x=np.array([np.nan,10.,np.nan,100.]).reshape(4,1,1)
        self.assertEqual(CleanDataset._fill_missing_air(x,'ffill').reshape(-1).tolist(),[0,10,10,100])
        filled=CleanDataset._fill_missing_air(x,'node_mean',2)
        self.assertEqual(filled[0,0,0],10)
        self.assertEqual(filled[2,0,0],10)

    def test_checkpoint_protocol_rejects_old_and_mismatched_models(self):
        expected={'version':'new','split':'622'}
        with self.assertRaisesRegex(ValueError,'no training protocol'):
            validate_checkpoint_protocol(SimpleNamespace(),expected)
        with self.assertRaisesRegex(ValueError,'split'):
            validate_checkpoint_protocol(SimpleNamespace(training_protocol={'version':'new','split':'811'}),expected)
        validate_checkpoint_protocol(SimpleNamespace(training_protocol=expected),expected)

    def test_early_stopping_is_exactly_ten_stale_epochs(self):
        self.assertFalse(should_early_stop(14,5,10))
        self.assertTrue(should_early_stop(15,5,10))

    def test_synthetic_two_stage_training_checkpoint_restore_and_evaluation(self):
        # Exercise the actual entry on synthetic data in /tmp. No real dataset
        # or historical result/checkpoint is read or overwritten.
        with tempfile.TemporaryDirectory(prefix='faststg-tex-',dir='/tmp') as temp:
            root=Path(temp)
            rng=np.random.default_rng(5)
            flow=rng.normal(10,2,(180,3,1)).astype(np.float32)
            flow[20,0,0]=np.nan
            flow[125,1,0]=np.nan
            np.save(root/'flow.npy',flow)
            np.save(root/'adj.npy',np.eye(3,dtype=np.float32))
            created=[]
            original=train.default_config
            def config_factory(*args,**kwargs):
                with patch('train.infer_dataset_meta',return_value={
                    'num_features':1,'num_vertices':3,'total_steps':180,
                    'points_per_hour':1,'steps_per_day':24}):
                    cfg=original(*args,**kwargs)
                cfg.data.feature_file=str(root/'flow.npy')
                cfg.data.spatial=str(root/'adj.npy')
                for attr,directory in (('PATH_MOD','model'),('PATH_LOG','log'),('PATH_FORECAST','forecast')):
                    setattr(cfg,attr,str(root/directory)+'/')
                    (root/directory).mkdir(exist_ok=True)
                created.append(cfg)
                return cfg
            options=['--paper_experiment','diagnostic','--data','AIR_BJ','--device','cpu','--epoch','2',
                     '--rf_ls_pretrain_epochs','1','--rf_steps','2',
                     '--n_samples','2','--is_test','true','--detailed_eval','true']
            with patch('train.default_config',side_effect=config_factory),patch('train.save2file'),contextlib.redirect_stdout(io.StringIO()):
                try:
                    cfg=params_for(*options,'--result_json',str(root/'train.json'))
                    train.main(cfg)
                    result=json.loads((root/'train.json').read_text())
                    selected=result['results'][0]
                    self.assertEqual(selected['checkpoint_stage'],'residual_flow')
                    self.assertEqual(selected['checkpoint_epoch'],2)
                    self.assertEqual(selected['checkpoint_selection_metric'],'crps')
                    model=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
                    head=torch.load(created[0].ls_pretrain_path,map_location='cpu',weights_only=False)
                    for key,value in head.location_scale_head.state_dict().items():
                        self.assertTrue(torch.equal(value,model.location_scale_head.state_dict()[key]))
                    train.main(params_for(*options,'--is_train','false','--ckpt_path',selected['checkpoint'],
                                          '--eval_warmup_batches','1','--result_json',str(root/'eval.json')))
                    evaluated=json.loads((root/'eval.json').read_text())['results'][0]
                    for key in ('mae','rmse','crps','mis','picp','interval_width'):
                        self.assertEqual(selected[key],evaluated[key])
                finally:
                    for cfg in created:
                        if cfg.logger.file:
                            cfg.logger.file.close()


if __name__=='__main__':
    unittest.main()
