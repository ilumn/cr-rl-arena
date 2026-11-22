"""Isolated one-tick vs bounded-resource-wait continuation experiment.

Both modes use decision-boundary MC with physical-tick discounting and decision-
state critic samples. This deliberately differs from the old all-tick critic
collector; compare these two branches, not just a new branch against old scores.
Use this runner for evaluation too: old evaluators do not implement held waits.
"""

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cr_sim
import torch

import train_gandalf as core
from gandalf import CARDS_PATH, CARD_NAMES
from gandalf_coordinate import CoordinateActorCritic, PlacementTemperature
from gandalf_coordinate_battle import (
    ROOT, atomic_save, coordinate_config, exclusive_lock, file_hash, load_turk, read_checkpoint,
)
from gandalf_entropy_battle import Tee
from gandalf_resource_wait import aggregate_stats, collect_resource_match, decision_returns


def resource_config():
    return {**coordinate_config(), 'decision_protocol': 'resource_wait_mc_v1'}


def setup(args, source, source_hash):
    if source.get('config') not in (coordinate_config(), resource_config()):
        raise ValueError('Source must be a coordinate battle checkpoint, not toy/original-head weights')
    parent = source.get('resource_experiment') or source['coordinate_experiment']
    settings = dict(parent['settings'])
    if settings.get('entropy_mode', 'joint') != 'joint':
        raise ValueError('Use a joint-entropy source to isolate this waiting experiment')
    files = ['gandalf_resource_battle.py', 'gandalf_resource_wait.py',
             'gandalf_coordinate_battle.py', 'gandalf_coordinate.py', 'gandalf_entropy_battle.py',
             'gandalf_entropy.py', 'model_brains.py', 'gandalf.py', 'train_gandalf.py',
             'train_gandalf_A.py', 'train_gandalf_rollout.py', 'theMechanicalTurk_1.0.py']
    experiment = dict(
        kind='resource_wait_battle_v1', source_sha256=source_hash, settings=settings,
        wait_mode=args.wait_mode, max_hold_ticks=math.ceil(args.max_hold_seconds * cr_sim.TICKS_PER_SECOND),
        threat_line=args.threat_line, run_seed=args.run_seed, threads=args.threads,
        estimator='mc', discount='per_physical_tick', critic_sampling='decision_states',
        training_placement_temperature=1., opponent_deck=parent['opponent_deck'],
        gandalf_deck=parent['gandalf_deck'], optimizer_transfer='all_parameters',
        code_hashes={n: file_hash(Path(__file__).parent/n) for n in files},
        simulator_sha256=file_hash(cr_sim.__file__), cards_sha256=file_hash(CARDS_PATH),
        torch_version=str(torch.__version__))
    return settings, experiment


def restore(args, source, settings, experiment):
    latest = args.run_dir/'latest.pt'
    if latest.exists():
        checkpoint, _ = read_checkpoint(latest)
        if (checkpoint.get('config') != resource_config()
                or checkpoint.get('resource_experiment') != experiment):
            raise ValueError('Run directory has different source/settings/code; use a new directory')
        reference = args.run_dir/'reference.pt'
        if not reference.exists() or file_hash(reference) != experiment['source_sha256']:
            raise ValueError('Frozen source is missing or changed')
    else:
        if args.run_dir.exists() and list(args.run_dir.glob('*.pt')):
            raise ValueError('Directory has checkpoints but no latest.pt; use a new directory')
        checkpoint = source
    torch.manual_seed(args.run_seed)
    model = CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if any(group['lr'] != settings['learning_rate'] for group in optimizer.param_groups):
        raise ValueError('Source Adam learning rate differs from recorded settings')
    completed = checkpoint['resource_updates'] if latest.exists() else 0
    progress = checkpoint['resource_progress'] if latest.exists() else dict(
        decisions=0, physical_ticks=0, matches=0)
    return model, optimizer, completed, checkpoint['next_seed'], progress


def collect(model, seed, seat, turk, experiment, opponent=None):
    return collect_resource_match(
        model, seed, opponent or turk.policy, turk.DECK, seat=seat,
        wait_mode=experiment['wait_mode'], max_hold_ticks=experiment['max_hold_ticks'],
        threat_line=experiment['threat_line'], gamma=experiment['settings']['gamma'])


def train(args):
    if args.source.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Keep source outside the trial directory')
    source, data = read_checkpoint(args.source)
    settings, experiment = setup(args, source, hashlib.sha256(data).hexdigest())
    turk = load_turk()
    if list(turk.DECK) != experiment['opponent_deck']:
        raise ValueError('MechanicalTurk deck differs from source')
    if args.dry_run:
        _, _, completed, _, progress = restore(args, source, settings, experiment)
        print(json.dumps(experiment, indent=2))
        print(f'Validated {completed} trial updates; experience={progress}; no files written')
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir/'.training.lock'), \
            (args.run_dir/'training.txt').open('a', buffering=1) as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        model, optimizer, completed, next_seed, progress = restore(args, source, settings, experiment)

        def snapshot():
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=resource_config(), next_seed=next_seed,
                        completed_batches=source['completed_batches']+completed,
                        prototype_updates=source['prototype_updates']+completed,
                        resource_updates=completed, resource_progress=dict(progress),
                        resource_experiment=experiment)

        if not (args.run_dir/'initial.pt').exists():
            (args.run_dir/'reference.pt').write_bytes(data)
            atomic_save(snapshot(), args.run_dir/'initial.pt')
            atomic_save(snapshot(), args.run_dir/'latest.pt')
            atomic_save(experiment, args.run_dir/'experiment.json', json_format=True)
        print(f'Resource wait={args.wait_mode}; {completed}/{args.batches} trial updates; '
              f"hold limit={experiment['max_hold_ticks']} ticks; threat line={args.threat_line}", flush=True)
        print('All source weights/Adam state retained. MC discounts elapsed physical ticks. '
              'Critic samples decision states in BOTH branches. Forced ticks have no PPO entries.', flush=True)
        with patch.dict(core.SETTINGS, settings):
            while completed < args.batches:
                batch_seed=args.run_seed+10_000*(source['completed_batches']+completed)
                torch.manual_seed(batch_seed)
                random.seed(batch_seed)
                records, outcomes, feedback, match_stats = [], Counter(), Counter(), []

                def opponent(obs):
                    if obs['tick'] > 0:
                        feedback[obs['last_action']] += 1
                    return turk.policy(obs)

                while len(records)<settings['target_choices'] or sum(outcomes.values())<settings['minimum_matches']:
                    match, result, stats=collect(model,next_seed,(next_seed-1)%2,turk,experiment,opponent)
                    next_seed+=1
                    records.extend(match)
                    outcomes[result]+=1
                    match_stats.append(stats)
                stats=aggregate_stats(match_stats)
                print(f'\nTrial update {completed+1}: {len(records)} decisions, '
                      f"{stats['counts']['physical_ticks']} physical ticks, results={dict(outcomes)}", flush=True)
                print('Resource diagnostics:', json.dumps(stats), flush=True)
                print('MechanicalTurk feedback:',dict(feedback))
                returns,advantages=decision_returns(records,settings['gamma'])
                core.report_rollout_quality(records,returns,advantages)
                core.report_exploration(model,records)
                core.ppo_update(model,optimizer,records,returns,advantages)
                completed+=1
                progress['decisions']+=len(records)
                progress['physical_ticks']+=stats['counts']['physical_ticks']
                progress['matches']+=sum(outcomes.values())
                atomic_save(snapshot(),args.run_dir/'latest.pt')
                if completed%5==0 or completed==args.batches:
                    atomic_save(snapshot(),args.run_dir/f'update_{completed}.pt')
                print(f'Saved trial update {completed}; experience={progress}',flush=True)


def evaluate(args):
    checkpoint,data=read_checkpoint(args.checkpoint)
    if checkpoint.get('config')!=resource_config():
        raise ValueError('Use a resource-wait trial checkpoint with this evaluator')
    experiment=checkpoint['resource_experiment']
    model=CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint['model'])
    policy=PlacementTemperature(model,args.temperature).eval()
    turk=load_turk()
    plan=dict(kind='resource_wait_evaluation_v1',checkpoint_sha256=hashlib.sha256(data).hexdigest(),
              temperature=args.temperature,games=args.games,seed_start=args.seed_start,
              wait_mode=experiment['wait_mode'],max_hold_ticks=experiment['max_hold_ticks'],
              threat_line=experiment['threat_line'],threads=args.threads,
              opponent_sha256=file_hash(Path(__file__).parent/'theMechanicalTurk_1.0.py'),
              simulator_sha256=file_hash(cr_sim.__file__),
              code_hashes={n:file_hash(Path(__file__).parent/n) for n in [
                  'gandalf_resource_battle.py','gandalf_resource_wait.py','gandalf_coordinate.py',
                  'model_brains.py','gandalf.py']})
    output=args.output or args.checkpoint.parent/f'evaluation_{args.checkpoint.stem}_{args.seed_start}_T{args.temperature:g}.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(output.with_suffix(output.suffix+'.lock')):
        report=dict(plan=plan,cases=[])
        if output.exists():
            report=json.loads(output.read_text())
            if ({k:v for k,v in report['plan'].items() if k!='games'}
                    !={k:v for k,v in plan.items() if k!='games'}):
                raise ValueError('Evaluation output belongs to different settings; choose --output')
            report['plan']=plan
        done={(c['seed'],c['seat']) for c in report['cases']}
        for seed in range(args.seed_start,args.seed_start+args.games//2):
            for seat in (0,1):
                if (seed,seat) in done:
                    continue
                torch.manual_seed(seed*2+seat)
                random.seed(seed*2+seat)
                _,result,stats=collect(policy,seed,seat,turk,experiment)
                report['cases'].append(dict(seed=seed,seat=seat,result=result,stats=stats))
                atomic_save(report,output,json_format=True)
                print(f"{experiment['wait_mode']}: completed seed={seed} seat={seat}: {result}",flush=True)
        cases=[c for c in report['cases'] if args.seed_start<=c['seed']<args.seed_start+args.games//2]
        results=Counter(c['result'] for c in cases)
        stats=aggregate_stats([c['stats'] for c in cases])
        report['summary']=dict(results=dict(results),win_rate=results['win']/len(cases),
                               seat_results=[dict(Counter(c['result'] for c in cases if c['seat']==s)) for s in (0,1)],
                               diagnostics=stats)
        atomic_save(report,output,json_format=True)
        print(f"{experiment['wait_mode']} T={args.temperature:g}: {results['win']}W/"
              f"{results['loss']}L/{results['draw']}D; win rate={results['win']/len(cases):.1%}",flush=True)
        print('Resource diagnostics:',json.dumps(stats),flush=True)
        print(f'Saved evaluation: {output}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    trainer=commands.add_parser('train')
    trainer.add_argument('--source',type=Path,required=True)
    trainer.add_argument('--wait-mode',choices=['tick','reserve'],required=True)
    trainer.add_argument('--run-dir',type=Path)
    trainer.add_argument('--batches',type=int,default=5,help='total updates in this new trial')
    trainer.add_argument('--run-seed',type=int,default=12345)
    trainer.add_argument('--max-hold-seconds',type=float,default=3.)
    trainer.add_argument('--threat-line',type=float,default=17.)
    trainer.add_argument('--dry-run',action='store_true')
    evaluator=commands.add_parser('evaluate')
    evaluator.add_argument('checkpoint',type=Path)
    evaluator.add_argument('--games',type=int,default=20)
    evaluator.add_argument('--temperature',type=float,default=.25)
    evaluator.add_argument('--seed-start',type=int,default=600_000)
    evaluator.add_argument('--output',type=Path)
    for command in (trainer,evaluator):
        command.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if args.threads<=0:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command=='train':
        if args.batches<0 or not 0<args.max_hold_seconds<float('inf') or not 0<=args.threat_line<=32:
            parser.error('invalid batch/hold/threat settings')
        args.run_dir=args.run_dir or ROOT/f'checkpoints/gandalf/resource_{args.wait_mode}_v1'
        train(args)
    else:
        if args.games<=0 or args.games%2 or args.seed_start<0 or not 0<args.temperature<float('inf'):
            parser.error('games must be positive/even; seed-start nonnegative; temperature finite/positive')
        evaluate(args)


if __name__=='__main__':
    main()
