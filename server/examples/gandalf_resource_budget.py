"""Resume resource-wait trials to a total decision budget, including prior updates.

Uses the unchanged resource runner one complete batch at a time. That preserves
existing code fingerprints, per-batch RNG seeds, optimizer state and MC episodes.
The first completed batch reaching the budget ends training; overshoot is reported.
"""

import argparse
import contextlib
import copy
import io
from pathlib import Path

import torch

import gandalf_resource_battle as battle


def state(path):
    return battle.read_checkpoint(path) if path.exists() else (None, None)


def run(args):
    train_args = copy.copy(args)
    train_args.batches = 0
    train_args.dry_run = True
    # Validate the original source, settings and resume fingerprints without
    # changing any files. Avoid printing the whole manifest on every invocation.
    with contextlib.redirect_stdout(io.StringIO()):
        battle.train(train_args)
    latest = args.run_dir / 'latest.pt'
    saved, _ = state(latest)
    current = saved['resource_progress']['decisions'] if saved else 0
    print(f'Validated {args.wait_mode}: {current} decisions already collected; '
          f'target={args.decision_budget} total decisions.', flush=True)
    if args.dry_run:
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    frozen = args.run_dir / f'budget_{args.decision_budget}.pt'
    with battle.exclusive_lock(args.run_dir / '.budget.lock'):
        saved, data = state(latest)
        if saved is None:
            train_args.dry_run = False
            battle.train(train_args)  # Initialize an untouched fork, zero updates.
            saved, data = state(latest)
        if frozen.exists():
            frozen_state, _ = battle.read_checkpoint(frozen)
            if (frozen_state.get('resource_experiment') != saved['resource_experiment']
                    or frozen_state['resource_progress']['decisions'] < args.decision_budget):
                raise ValueError('Existing budget snapshot belongs to a different/incomplete experiment')
            print(f'Budget snapshot already exists: {frozen}; '
                  f"decisions={frozen_state['resource_progress']['decisions']}", flush=True)
            return
        while saved['resource_progress']['decisions'] < args.decision_budget:
            previous = saved['resource_progress']['decisions']
            train_args.batches = saved['resource_updates'] + 1
            train_args.dry_run = False
            battle.train(train_args)
            saved, data = state(latest)
            current = saved['resource_progress']['decisions']
            if current <= previous:
                raise RuntimeError('Training did not advance the decision counter')
            print(f'Budget progress: {current}/{args.decision_budget} decisions', flush=True)
        # Preserve the exact completed checkpoint bytes for stable evaluation.
        # Exclusive creation prevents overwriting any existing budget snapshot.
        with frozen.open('xb') as handle:
            handle.write(data)
        actual = saved['resource_progress']['decisions']
        report = dict(kind='resource_decision_budget_v1', target_decisions=args.decision_budget,
                      actual_decisions=actual, overshoot=actual-args.decision_budget,
                      resource_updates=saved['resource_updates'], wait_mode=args.wait_mode,
                      source_sha256=saved['resource_experiment']['source_sha256'],
                      wrapper_sha256=battle.file_hash(Path(__file__)),
                      checkpoint_sha256=battle.file_hash(frozen),
                      checkpoint=str(frozen), progress=saved['resource_progress'])
        battle.atomic_save(report, frozen.with_suffix('.json'), json_format=True)
        print(f'Budget complete: {actual} decisions; overshoot={actual-args.decision_budget}. '
              f'Saved {frozen}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--wait-mode', choices=['tick', 'reserve'], required=True)
    parser.add_argument('--decision-budget', type=int, default=25_000)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--run-seed', type=int, default=12345)
    parser.add_argument('--max-hold-seconds', type=float, default=3.)
    parser.add_argument('--threat-line', type=float, default=17.)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (args.decision_budget <= 0 or args.threads <= 0
            or not 0 < args.max_hold_seconds < float('inf')
            or not 0 <= args.threat_line <= 32):
        parser.error('Invalid decision budget, thread, hold or threat settings')
    torch.set_num_threads(args.threads)
    args.run_dir = args.run_dir or battle.ROOT / f'checkpoints/gandalf/resource_{args.wait_mode}_v1'
    run(args)


if __name__ == '__main__':
    main()
