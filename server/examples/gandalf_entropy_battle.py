"""Fork a coordinate battle checkpoint for joint vs separated entropy trials.

Both branches retain ALL source weights and Adam state. Only the entropy rule
changes. Existing coordinate runs, source files, and MechanicalTurk are untouched.
Trial update numbers start at zero; prototype/source batch counters are retained.
"""

import argparse
import contextlib
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cr_sim
import torch

import train_gandalf as core
from gandalf import CARDS_PATH, CARD_NAMES
from gandalf_coordinate import CoordinateActorCritic
from gandalf_coordinate_battle import (
    ROOT, atomic_save, coordinate_config, evaluate, exclusive_lock,
    file_hash, load_turk, read_checkpoint,
)
from gandalf_entropy import separated_evaluator


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup(args, source, source_hash):
    if source.get('config') != coordinate_config():
        raise ValueError('Source must be a coordinate BATTLE checkpoint, not toy/original weights')
    settings = dict(source['coordinate_experiment']['settings'])
    previous_coefficient = settings['entropy_coefficient']
    slot = previous_coefficient if args.slot_entropy is None else args.slot_entropy
    tile = previous_coefficient if args.tile_entropy is None else args.tile_entropy
    if args.entropy_mode == 'joint' and slot != tile:
        raise ValueError('Joint mode uses one coefficient; slot/tile coefficients must match')
    # Existing PPO multiplies its entropy output by this scalar. The separated
    # adapter accounts for it internally, allowing independent slot/tile betas.
    scale = slot if args.entropy_mode == 'joint' else (previous_coefficient or 1.)
    settings.update(entropy_coefficient=scale, entropy_mode=args.entropy_mode,
                    slot_entropy_coefficient=slot, tile_entropy_coefficient=tile)
    files = ['gandalf_entropy_battle.py', 'gandalf_entropy.py',
             'gandalf_coordinate_battle.py', 'gandalf_coordinate.py',
             'model_brains.py', 'gandalf.py', 'train_gandalf.py',
             'train_gandalf_rollout.py', 'train_gandalf_A.py', 'theMechanicalTurk_1.0.py']
    parent = source['coordinate_experiment']
    experiment = dict(
        kind='entropy_battle_ablation_v1', source_sha256=source_hash,
        source_completed_batches=source['completed_batches'],
        source_prototype_updates=source['prototype_updates'],
        settings=settings, run_seed=args.run_seed, threads=args.threads,
        training_placement_temperature=1., estimator='mc', gae_lambda=1.,
        opponent_deck=parent['opponent_deck'], gandalf_deck=parent['gandalf_deck'],
        optimizer_transfer='all_parameters',
        code_hashes={name: file_hash(Path(__file__).parent / name) for name in files},
        simulator_sha256=file_hash(cr_sim.__file__), cards_sha256=file_hash(CARDS_PATH),
        torch_version=str(torch.__version__))
    return settings, experiment


def restore(args, source, settings, experiment):
    latest = args.run_dir / 'latest.pt'
    if latest.exists():
        checkpoint, _ = read_checkpoint(latest)
        if (checkpoint.get('config') != coordinate_config()
                or checkpoint.get('coordinate_experiment') != experiment):
            raise ValueError('Run directory has a different source/settings/code; use a new directory')
        reference = args.run_dir / 'reference.pt'
        if not reference.exists() or file_hash(reference) != experiment['source_sha256']:
            raise ValueError('Frozen reference is missing or changed')
    else:
        if args.run_dir.exists() and list(args.run_dir.glob('*.pt')):
            raise ValueError('Directory contains checkpoints but no latest.pt; use a new directory')
        checkpoint = source
    torch.manual_seed(args.run_seed)
    model = CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if any(group['lr'] != settings['learning_rate'] for group in optimizer.param_groups):
        raise ValueError('Source Adam learning rate differs from recorded training settings')
    completed = checkpoint['entropy_updates'] if latest.exists() else 0
    progress = checkpoint['entropy_progress'] if latest.exists() else dict(
        meaningful_choices=0, simulator_ticks=0, completed_matches=0)
    return model, optimizer, completed, checkpoint['next_seed'], progress


def train(args):
    if args.source.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Keep the source outside the new trial directory')
    source, data = read_checkpoint(args.source)
    settings, experiment = setup(args, source, hashlib.sha256(data).hexdigest())
    turk = load_turk()
    if list(turk.DECK) != experiment['opponent_deck']:
        raise ValueError('MechanicalTurk deck differs from the source experiment')
    if args.dry_run:
        _, _, completed, _, progress = restore(args, source, settings, experiment)
        print(json.dumps(experiment, indent=2))
        print(f'Validated {completed} trial updates; experience={progress}; no files written')
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir / '.training.lock'), \
            (args.run_dir / 'training.txt').open('a', buffering=1) as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        model, optimizer, completed, next_seed, progress = restore(
            args, source, settings, experiment)

        def snapshot():
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=coordinate_config(), next_seed=next_seed,
                        completed_batches=source['completed_batches'] + completed,
                        prototype_updates=source['prototype_updates'] + completed,
                        entropy_updates=completed, entropy_progress=dict(progress),
                        coordinate_experiment=experiment)

        if not (args.run_dir / 'initial.pt').exists():
            (args.run_dir / 'reference.pt').write_bytes(data)
            atomic_save(snapshot(), args.run_dir / 'initial.pt')
            atomic_save(snapshot(), args.run_dir / 'latest.pt')
            atomic_save(experiment, args.run_dir / 'experiment.json', json_format=True)
        print(f'Entropy mode={args.entropy_mode}; {completed}/{args.batches} trial updates; '
              f"slot beta={settings['slot_entropy_coefficient']}, "
              f"tile beta={settings['tile_entropy_coefficient']}", flush=True)
        print('All source weights/Adam state retained; training temperature=1.0. '
              'Experience counts include whole-match rollout overshoot.', flush=True)
        evaluator = (core.evaluate_actions if args.entropy_mode == 'joint' else
                     separated_evaluator(settings['slot_entropy_coefficient'],
                                         settings['tile_entropy_coefficient'],
                                         settings['entropy_coefficient']))
        with patch.dict(core.SETTINGS, settings), patch.object(core, 'evaluate_actions', evaluator):
            while completed < args.batches:
                seed = args.run_seed + 10_000 * (source['completed_batches'] + completed)
                random.seed(seed)
                torch.manual_seed(seed)
                records, outcomes, feedback = [], Counter(), Counter()
                choices = 0

                def opponent(obs):
                    if obs['tick'] > 0:
                        feedback[obs['last_action']] += 1
                    return turk.policy(obs)

                while choices < settings['target_choices'] or sum(outcomes.values()) < settings['minimum_matches']:
                    match, result = core.collect_match(
                        model, next_seed, opponent_fn=opponent,
                        seat=(next_seed - 1) % 2, opponent_deck=turk.DECK)
                    next_seed += 1
                    records.extend(match)
                    choices += sum(sum(r['slot_mask']) > 1 for r in match)
                    outcomes[result] += 1
                print(f'\nTrial update {completed + 1}: {len(records)} ticks, '
                      f'{choices} choices, results={dict(outcomes)}', flush=True)
                print('MechanicalTurk feedback:', dict(feedback))
                returns, advantages = core.calculate_returns_and_advantages(
                    records, gamma=settings['gamma'], gae_lambda=1.)
                core.report_rollout_quality(records, returns, advantages)
                core.report_exploration(model, records)
                core.ppo_update(model, optimizer, records, returns, advantages)
                completed += 1
                progress['meaningful_choices'] += choices
                progress['simulator_ticks'] += len(records)
                progress['completed_matches'] += sum(outcomes.values())
                atomic_save(snapshot(), args.run_dir / 'latest.pt')
                if completed % 5 == 0 or completed == args.batches:
                    atomic_save(snapshot(), args.run_dir / f'update_{completed}.pt')
                print(f'Saved trial update {completed}; experience={progress}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--source', type=Path, required=True)
    trainer.add_argument('--entropy-mode', choices=['joint', 'separated'], required=True)
    trainer.add_argument('--run-dir', type=Path)
    trainer.add_argument('--batches', type=int, default=5, help='total trial updates, starting from zero')
    trainer.add_argument('--run-seed', type=int, default=12345)
    trainer.add_argument('--slot-entropy', type=float)
    trainer.add_argument('--tile-entropy', type=float)
    trainer.add_argument('--dry-run', action='store_true')
    evaluator = commands.add_parser('evaluate')
    evaluator.add_argument('checkpoint', type=Path)
    evaluator.add_argument('--games', type=int, default=20)
    evaluator.add_argument('--temperatures', type=float, nargs='+', default=[.25])
    evaluator.add_argument('--seed-start', type=int, default=500_000)
    evaluator.add_argument('--reference', type=Path)
    evaluator.add_argument('--no-reference', action='store_true')
    evaluator.add_argument('--output', type=Path)
    for command in (trainer, evaluator):
        command.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'train':
        if args.batches < 0 or any(v is not None and not 0 <= v < float('inf')
                                  for v in [args.slot_entropy, args.tile_entropy]):
            parser.error('batches must be nonnegative; entropy coefficients finite/nonnegative')
        args.run_dir = args.run_dir or ROOT / f'checkpoints/gandalf/entropy_{args.entropy_mode}_v1'
        train(args)
    else:
        if args.games <= 0 or args.games % 2 or args.seed_start < 0:
            parser.error('games must be positive/even; seed-start nonnegative')
        if (not all(0 < t < float('inf') for t in args.temperatures)
                or len(set(args.temperatures)) != len(args.temperatures)):
            parser.error('temperatures must be distinct, finite and positive')
        evaluate(args)


if __name__ == '__main__':
    main()
