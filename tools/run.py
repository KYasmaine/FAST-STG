#!/usr/bin/env python3
"""Build or execute fixed-manuscript experiments, without reusing old results.

Commands are printed unless --execute is given. Efficiency/evaluation require
an explicitly supplied checkpoint carrying the current training protocol.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import PAPER_VARIANTS


def build_commands(args):
    output = Path(args.output_dir).resolve()
    base = [sys.executable, str(ROOT/'train.py'), '--paper_profile','www2027',
            '--data',args.data,'--device',args.device,'--seed',str(args.seed),
            '--lr','0.002','--batch_size','8','--epoch','300','--early_stop','10',
            '--T_h','12','--T_p','12','--hidden_size','32','--rf_ls_hidden','128',
            '--split_ratio','622','--detailed_eval','true']
    commands = []
    def add(name, options=(), checkpoint=False):
        command = base + list(options) + ['--result_json',str(output/(name+'.json'))]
        if checkpoint:
            if not args.checkpoint:
                raise ValueError(f'{args.task} requires --checkpoint from the current protocol')
            command += ['--is_train','false','--ckpt_path',str(Path(args.checkpoint).resolve())]
        commands.append(command)
    if args.task == 'main':
        add(f'{args.data.lower()}_main_seed{args.seed}')
    elif args.task == 'ablation':
        for variant in PAPER_VARIANTS[1:]:
            add(f'{args.data.lower()}_{variant}_seed{args.seed}', ['--paper_variant',variant])
    elif args.task in ('evaluate','efficiency'):
        options = ['--n_samples','100','--rf_steps','20']
        if args.task == 'efficiency':
            options += ['--detailed_eval','false','--forecast_save_limit','1','--eval_warmup_batches','3']
        add(f'{args.data.lower()}_{args.task}_seed{args.seed}', options, checkpoint=True)
    elif args.task == 'sensitivity':
        base += ['--paper_experiment', 'sensitivity']
        if args.data != 'AIR_GZ':
            raise ValueError('The manuscript sensitivity study uses AIR_GZ')
        # K, gamma, S affect inference only; reuse the selected main checkpoint.
        for key,flag,values in (
            ('k','--rf_steps',(5,10,20,30)),
            ('gamma','--rf_eval_sigma',(.02,.05,.08,.10,.15)),
            ('s','--n_samples',(8,20,50,100,200)),
        ):
            for value in values:
                suffix=f'{value:.2f}'.replace('.','') if key=='gamma' else str(value)
                add(f'air_gz_hparam_{key}{suffix}_split622_s100_seed{args.seed}' if key!='s'
                    else f'air_gz_hparam_s{suffix}_split622_seed{args.seed}',
                    [flag,str(value)], checkpoint=True)
        for epochs in (1,3,5,10):
            add(f'air_gz_hparam_els{epochs}_split622_s100_seed{args.seed}',
                ['--rf_ls_pretrain_epochs',str(epochs)])
        add(f'air_gz_split622_location_scale_flow_gamma008_s100_eval_seed{args.seed}',checkpoint=True)
    return commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('main','ablation','evaluate','sensitivity','efficiency'),default='main')
    parser.add_argument('--data',choices=('AIR_BJ','AIR_GZ','PEMS03','PEMS04','PEMS07','PEMS08'),default='PEMS08')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=2022)
    parser.add_argument('--checkpoint',default='')
    parser.add_argument('--output_dir',default=str(ROOT/'output/metrics/www2027'))
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    commands=build_commands(args)
    for command in commands:
        print(shlex.join(command),flush=True)
        if args.execute:
            subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':
    main()
