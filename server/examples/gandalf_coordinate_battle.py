"""Isolated coordinate-head battle fork, continuation and temperature evaluation.

Training reuses normal terminal rewards, masks, sampler and PPO at temperature 1.
The source must be a real original-head battle checkpoint, never toy weights.
"""

import argparse
import copy
import fcntl
import hashlib
import io
import json
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import cr_sim
import torch

from evaluate_gandalf import load_turk, play_match
from gandalf import CARD_NAMES, DECK, CARDS_PATH
from gandalf_coordinate import CoordinateActorCritic, PlacementTemperature
from model_brains import ActorCritic
from train_gandalf_A import SETTINGS as ACCEPTED_SETTINGS
from train_gandalf import (
    SETTINGS, collect_match, calculate_returns_and_advantages, ppo_update,
    report_exploration, report_rollout_quality,
)
from train_gandalf_rollout import current_config, file_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / 'checkpoints/gandalf/coordinate_turk_v1'


def coordinate_config():
    return {**current_config(), 'architecture': 'entity_pool_coordinate_tile_v1',
            'coordinate_hidden_size': 32}


def read_checkpoint(path):
    # Hash and deserialize exactly the same bytes, even if latest.pt is replaced.
    data = path.read_bytes()
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=True), data


def atomic_save(value, path, json_format=False):
    temporary = path.with_suffix(path.suffix + '.tmp')
    if json_format:
        temporary.write_text(json.dumps(value, indent=2) + '\n')
    else:
        torch.save(value, temporary)
    temporary.replace(path)


@contextmanager
def exclusive_lock(path):
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {path}') from None
        yield


def fork_model(source, seed, settings):
    """Retain common weights/Adam moments; initialize only the new tile head."""
    if source.get('config') != current_config():
        raise ValueError('Source must be a compatible original-head battle checkpoint')
    torch.manual_seed(seed)
    original = ActorCritic(len(CARD_NAMES))
    original.load_state_dict(source['model'])
    # Independent, reproducible initialization for the fresh placement head.
    torch.manual_seed(seed)
    model = CoordinateActorCritic(len(CARD_NAMES))
    common = {k: v for k, v in source['model'].items() if not k.startswith('tile_head.')}
    missing = model.load_state_dict(common, strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith('tile_head.') for k in missing.missing_keys)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in common.items())

    old_optimizer = torch.optim.Adam(original.parameters(), lr=settings['learning_rate'])
    old_optimizer.load_state_dict(source['optimizer'])
    if len(old_optimizer.param_groups) != 1:
        raise ValueError('Expected a single Adam parameter group in the source')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    optimizer.param_groups[0].update({
        k: copy.deepcopy(v) for k, v in old_optimizer.param_groups[0].items()
        if k not in ('params', 'param_names', 'lr')})
    original_parameters = dict(original.named_parameters())
    for name, parameter in model.named_parameters():
        if not name.startswith('tile_head.'):
            old_parameter = original_parameters[name]
            if old_parameter in old_optimizer.state:
                optimizer.state[parameter] = copy.deepcopy(old_optimizer.state[old_parameter])
    return model, optimizer


def experiment_settings(args, source_hash, turk):
    settings = dict(ACCEPTED_SETTINGS)
    if args.target_choices is not None:
        settings['target_choices'] = args.target_choices
    if args.minimum_matches is not None:
        settings['minimum_matches'] = args.minimum_matches
    files = ['gandalf_coordinate_battle.py', 'gandalf_coordinate.py', 'gandalf.py',
             'model_brains.py', 'train_gandalf.py', 'train_gandalf_A.py',
             'train_gandalf_rollout.py', 'theMechanicalTurk_1.0.py']
    experiment = dict(
        kind='coordinate_battle_v1', source_sha256=source_hash,
        settings=settings, run_seed=args.run_seed, threads=args.threads,
        estimator='mc', gae_lambda=1., training_placement_temperature=1.,
        opponent_deck=list(turk.DECK), gandalf_deck=list(DECK),
        optimizer_transfer='common_parameters_only',
        code_hashes={name: file_hash(Path(__file__).parent / name) for name in files},
        cards_sha256=file_hash(CARDS_PATH), simulator_sha256=file_hash(cr_sim.__file__),
        torch_version=str(torch.__version__))
    return settings, experiment


def restore_or_fork(args, source, settings, experiment):
    latest = args.run_dir / 'latest.pt'
    if latest.exists():
        checkpoint, _ = read_checkpoint(latest)
        if (checkpoint.get('config') != coordinate_config()
                or checkpoint.get('coordinate_experiment') != experiment):
            raise ValueError('Run directory has different code/settings/source; use a new directory')
        reference = args.run_dir / 'reference.pt'
        if not reference.exists() or file_hash(reference) != experiment['source_sha256']:
            raise ValueError('Frozen reference is missing or differs from the original source')
        model = CoordinateActorCritic(len(CARD_NAMES))
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['prototype_updates'], checkpoint['next_seed']
    if args.run_dir.exists() and list(args.run_dir.glob('*.pt')):
        raise ValueError('Directory contains checkpoints without latest.pt; use a new directory')
    model, optimizer = fork_model(source, args.run_seed, settings)
    return model, optimizer, 0, source['next_seed']


def train(args):
    if args.source.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Source must be outside the prototype directory')
    source, source_bytes = read_checkpoint(args.source)
    if source.get('config') != current_config():
        raise ValueError('Expected a real original-head battle checkpoint as source')
    turk = load_turk()
    settings, experiment = experiment_settings(
        args, hashlib.sha256(source_bytes).hexdigest(), turk)
    if args.dry_run:
        _, _, completed, _ = restore_or_fork(args, source, settings, experiment)
        print(json.dumps(experiment, indent=2))
        print(f'Validated: {completed} prototype batches completed; no files written')
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir / '.training.lock'):
        model, optimizer, completed, next_seed = restore_or_fork(
            args, source, settings, experiment)
        reference = args.run_dir / 'reference.pt'
        if not reference.exists():
            reference.write_bytes(source_bytes)
        base_batch = source['completed_batches']

        def snapshot():
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=coordinate_config(), next_seed=next_seed,
                        completed_batches=base_batch + completed,
                        prototype_updates=completed, coordinate_experiment=experiment)

        if not (args.run_dir / 'initial.pt').exists():
            atomic_save(snapshot(), args.run_dir / 'initial.pt')
            atomic_save(snapshot(), args.run_dir / 'latest.pt')
            atomic_save(experiment, args.run_dir / 'experiment.json', json_format=True)
        print(f'Coordinate fork from source batch {base_batch}; '
              f'{completed}/{args.batches} prototype batches completed', flush=True)
        print('Fresh tile head; common weights and Adam state retained. '
              'Training placement temperature = 1.0.', flush=True)
        with patch.dict(SETTINGS, settings):
            while completed < args.batches:
                batch_seed = args.run_seed + 10_000 * (base_batch + completed)
                random.seed(batch_seed)
                torch.manual_seed(batch_seed)
                batch, outcomes, opponent_feedback = [], Counter(), Counter()
                choices = 0

                def opponent(obs):
                    if obs['tick'] > 0:
                        opponent_feedback[obs['last_action']] += 1
                    return turk.policy(obs)

                while choices < settings['target_choices'] or sum(outcomes.values()) < settings['minimum_matches']:
                    records, result = collect_match(
                        model, next_seed, opponent_fn=opponent,
                        seat=(next_seed - 1) % 2, opponent_deck=turk.DECK)
                    next_seed += 1
                    batch.extend(records)
                    choices += sum(sum(r['slot_mask']) > 1 for r in records)
                    outcomes[result] += 1
                print(f'\nPrototype batch {completed + 1}: {len(batch)} ticks, '
                      f'{choices} choices, results={dict(outcomes)}', flush=True)
                print('MechanicalTurk feedback:', dict(opponent_feedback))
                returns, advantages = calculate_returns_and_advantages(
                    batch, gamma=settings['gamma'], gae_lambda=1.)
                report_rollout_quality(batch, returns, advantages)
                report_exploration(model, batch)
                ppo_update(model, optimizer, batch, returns, advantages)
                completed += 1
                atomic_save(snapshot(), args.run_dir / 'latest.pt')
                if completed % 5 == 0 or completed == args.batches:
                    atomic_save(snapshot(), args.run_dir / f'update_{completed}.pt')
                print(f'Saved prototype batch {completed} in {args.run_dir}', flush=True)


def load_battle_model(checkpoint):
    config = checkpoint.get('config')
    if config == current_config():
        model = ActorCritic(len(CARD_NAMES))
    elif config == coordinate_config():
        model = CoordinateActorCritic(len(CARD_NAMES))
    else:
        raise ValueError('Incompatible battle architecture/configuration; toy checkpoints are not accepted')
    model.load_state_dict(checkpoint['model'])
    return model.eval()


def evaluate(args):
    checkpoint, data = read_checkpoint(args.checkpoint)
    if checkpoint.get('config') != coordinate_config():
        raise ValueError('Evaluate a coordinate battle checkpoint, not toy or original weights')
    model = load_battle_model(checkpoint)
    turk = load_turk()
    policies = [(f'T={t:g}', PlacementTemperature(model, t).eval()) for t in args.temperatures]
    reference = args.reference or args.checkpoint.parent / 'reference.pt'
    reference_hash = None
    if not args.no_reference:
        original, reference_bytes = read_checkpoint(reference)
        reference_hash = hashlib.sha256(reference_bytes).hexdigest()
        expected = checkpoint['coordinate_experiment']['source_sha256']
        if reference_hash != expected:
            raise ValueError('Evaluation reference is not the frozen training source')
        policies.insert(0, ('reference', load_battle_model(original)))
    plan = dict(kind='coordinate_battle_evaluation_v1',
                checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                reference_sha256=reference_hash, games=args.games,
                seed_start=args.seed_start, temperatures=args.temperatures,
                opponent_sha256=file_hash(Path(__file__).parent / 'theMechanicalTurk_1.0.py'),
                simulator_sha256=file_hash(cr_sim.__file__),
                evaluation_code_sha256=file_hash(Path(__file__).parent / 'evaluate_gandalf.py'),
                coordinate_code_sha256=file_hash(Path(__file__).parent / 'gandalf_coordinate.py'),
                runner_sha256=file_hash(Path(__file__)), threads=args.threads)
    output = args.output or args.checkpoint.parent / (
        f'evaluation_{args.checkpoint.stem}_{args.seed_start}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output.with_suffix(output.suffix + '.lock')):
        report = dict(plan=plan, cases=[])
        if output.exists():
            report = json.loads(output.read_text())
            old_plan = report.get('plan', {})
            if ({k: v for k, v in old_plan.items() if k != 'games'}
                    != {k: v for k, v in plan.items() if k != 'games'}):
                raise ValueError('Evaluation output belongs to different settings; choose --output')
            # Increasing --games adds new paired openings without replaying old cases.
            report['plan'] = plan
        done = {(c['policy'], c['seed'], c['seat']) for c in report['cases']}
        for label, policy in policies:
            for seed in range(args.seed_start, args.seed_start + args.games // 2):
                for seat in (0, 1):
                    if (label, seed, seat) in done:
                        continue
                    result, statuses = play_match(model=policy, seed=seed, seat=seat,
                                                  opponent_policy=turk.policy, opponent_deck=turk.DECK)
                    report['cases'].append(dict(policy=label, seed=seed, seat=seat,
                                                result=result, deployment_statuses=dict(statuses)))
                    atomic_save(report, output, json_format=True)
                    print(f'{label}: completed '
                          f"{sum(c['policy'] == label for c in report['cases'])}/{args.games}", flush=True)
        report['summary'] = {}
        for label, _ in policies:
            cases = [c for c in report['cases'] if c['policy'] == label
                     and args.seed_start <= c['seed'] < args.seed_start + args.games // 2]
            results, statuses = Counter(c['result'] for c in cases), Counter()
            for case in cases:
                statuses.update(case['deployment_statuses'])
            summary = dict(results=dict(results), win_rate=results['win'] / len(cases),
                           seat_results=[dict(Counter(c['result'] for c in cases if c['seat'] == s))
                                         for s in (0, 1)], deployment_statuses=dict(statuses))
            report['summary'][label] = summary
            print(f"{label}: {results['win']}W/{results['loss']}L/{results['draw']}D, "
                  f"win rate={summary['win_rate']:.1%}, rejected="
                  f"{sum(v for k,v in statuses.items() if k != 'ok')}", flush=True)
        atomic_save(report, output, json_format=True)
    print(f'Saved evaluation: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    trainer = subparsers.add_parser('train', help='fork/resume isolated battle training')
    trainer.add_argument('--source', type=Path, required=True)
    trainer.add_argument('--run-dir', type=Path, default=DEFAULT_DIR)
    trainer.add_argument('--batches', type=int, default=5, help='total prototype batches, not source batch numbers')
    trainer.add_argument('--run-seed', type=int, default=12345)
    trainer.add_argument('--target-choices', type=int)
    trainer.add_argument('--minimum-matches', type=int)
    trainer.add_argument('--dry-run', action='store_true')
    evaluator = subparsers.add_parser('evaluate', help='paired MechanicalTurk evaluation; resumes saved cases')
    evaluator.add_argument('checkpoint', type=Path)
    evaluator.add_argument('--games', type=int, default=20, help='games per policy, balanced seats')
    evaluator.add_argument('--temperatures', type=float, nargs='+', default=[1., .25])
    evaluator.add_argument('--seed-start', type=int, default=500_000)
    evaluator.add_argument('--reference', type=Path)
    evaluator.add_argument('--no-reference', action='store_true')
    evaluator.add_argument('--output', type=Path)
    for command in (trainer, evaluator):
        command.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        if args.batches < 0 or any(v is not None and v <= 0 for v in [args.target_choices, args.minimum_matches]):
            parser.error('batches must be nonnegative; rollout settings must be positive')
        train(args)
    else:
        if args.games <= 0 or args.games % 2 or args.seed_start < 0:
            parser.error('games must be positive/even and seed-start nonnegative')
        if not all(0 < t < float('inf') for t in args.temperatures):
            parser.error('temperatures must be finite and positive')
        if len(set(args.temperatures)) != len(args.temperatures):
            parser.error('temperatures must be distinct')
        evaluate(args)


if __name__ == '__main__':
    main()
