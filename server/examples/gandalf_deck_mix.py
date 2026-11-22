"""Original/Wizard deck continuation with per-deck budgets and paired evaluation.

Wizard replaces Musketeer. The network/vocabulary/masks are unchanged; own_deck
is passed explicitly to the simulator. Old runners/global DECK are never mutated.
Both decks use one-tick waits and decision-state MC with physical-tick discounts.
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
from gandalf import (
    CARD_INDEX, CARD_NAMES, CARDS_PATH, DECK, action_masks, decode_action, encode_values,
)
from gandalf_coordinate import CoordinateActorCritic, PlacementTemperature
from gandalf_coordinate_battle import ROOT, atomic_save, exclusive_lock, file_hash, load_turk, read_checkpoint
from gandalf_entropy_battle import Tee
from gandalf_resource_battle import resource_config
from gandalf_resource_wait import aggregate_stats, decision_returns
from model_brains import select_action


DECKS = {
    'original': list(DECK),
    'wizard': ['wizard' if card == 'musketeer' else card for card in DECK],
}
DEFAULT_DIR = ROOT / 'checkpoints/gandalf/deck_mix_wizard_v1'


def mixed_config():
    config = resource_config()
    config.pop('deck')
    return {**config, 'deck_pool': DECKS, 'deck_protocol': 'original_wizard_mix_v1'}


def validate_decks(opponent_deck):
    for name, deck in DECKS.items():
        if len(deck) != 8 or len(set(deck)) != 8 or any(c not in CARD_INDEX for c in deck):
            raise ValueError(f'Invalid deck: {name}')
        for seat in (0, 1):
            battle = cr_sim.Battle()
            decks = [None, None]
            decks[seat], decks[1-seat] = deck, opponent_deck
            battle.reset(seed=42, deck0=decks[0], deck1=decks[1])
            action_masks(battle.view(seat).observe())


@torch.no_grad()
def collect_match(model, seed, seat, own_deck, opponent_fn, opponent_deck, gamma):
    """Same tick-mode decision protocol as resource_wait, with an explicit deck."""
    battle = cr_sim.Battle()
    decks = [None, None]
    decks[seat], decks[1-seat] = list(own_deck), list(opponent_deck)
    battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])
    view, other = battle.view(seat), battle.view(1-seat)
    records, cards, deployments, counts = [], Counter(), Counter(), Counter()
    in_hand, affordable, deployable = Counter(), Counter(), Counter()
    elixir_sum = 0.
    while not battle.done():
        obs = view.observe()
        action, selected = None, None
        if not any(obs['legal']):
            counts['forced_unaffordable_ticks'] += 1
        else:
            slots, tiles = action_masks(obs)
            if sum(slots) > 1:
                encoded = encode_values(obs)
                selected, tile, log_prob, value = select_action(model, encoded, slots, tiles)
                action = decode_action(selected, tile)
                records.append(dict(observations=encoded, slot_action=selected,
                                    tile_id=tile, slot_mask=slots, tile_masks=tiles,
                                    log_probability=log_prob, value_estimate=value,
                                    start_tick=obs['tick'], duration_ticks=0, reward=0., done=False))
                counts['decisions'] += 1
                counts['multi_card_decisions'] += sum(obs['legal']) >= 2
                counts['voluntary_waits'] += selected == 0
                elixir_sum += obs['elixir']
                in_hand.update(obs['hand'])
                affordable.update(card for i, card in enumerate(obs['hand']) if obs['legal'][i])
                deployable.update(card for i, card in enumerate(obs['hand']) if slots[i+1])
            else:
                counts['forced_no_placement_ticks'] += 1
        opponent_action = opponent_fn(other.observe())
        pair = [None, None]
        pair[seat] = action
        pair[1-seat] = tuple(opponent_action) if opponent_action is not None else None
        battle.step(*pair)
        after = view.observe()
        counts['physical_ticks'] += 1
        if selected:
            status = after['last_action']
            deployments[status] += 1
            if status == 'ok':
                cards[obs['hand'][selected-1]] += 1
        if records:
            record = records[-1]
            record['duration_ticks'] = int(after['tick']-record['start_tick'])
            record['done'] = battle.done()
            if record['done']:
                reward = {'win': 1., 'draw': 0., 'loss': -1.}[view.result()]
                record['reward'] = gamma**(record['duration_ticks']-1)*reward
    if not records:
        raise RuntimeError('Match ended without a decision opportunity')
    assert records[-1]['done'] and sum(r['done'] for r in records) == 1
    assert all(r['duration_ticks'] >= 1 and sum(r['slot_mask']) > 1 for r in records)
    stats = dict(counts=dict(counts), holds={}, cards_played=dict(cards),
                 deployment_statuses=dict(deployments), decision_elixir_sum=elixir_sum,
                 card_opportunities={card: dict(in_hand=in_hand[card], affordable=affordable[card],
                                                 deployable=deployable[card]) for card in own_deck})
    return records, view.result(), stats


def merge_stats(stats):
    combined = aggregate_stats(stats)
    opportunities = {}
    for entry in stats:
        for card, counts in entry.get('card_opportunities', {}).items():
            target = opportunities.setdefault(card, Counter())
            target.update(counts)
    combined['card_opportunities'] = {card: dict(counts) for card, counts in opportunities.items()}
    return combined


def empty_progress():
    return {name: dict(decisions=0, matches=0, seat_matches=[0, 0], results={}, stats=merge_stats([]))
            for name in DECKS}


def setup(args, source, source_hash, turk):
    if (source.get('config') != resource_config()
            or source.get('resource_experiment', {}).get('wait_mode') != 'tick'):
        raise ValueError('Source must be a one-tick coordinate resource BATTLE checkpoint')
    parent = source['resource_experiment']
    settings = dict(parent['settings'])
    if settings.get('entropy_mode', 'joint') != 'joint':
        raise ValueError('Use the joint-entropy baseline for this deck experiment')
    if list(turk.DECK) != parent['opponent_deck']:
        raise ValueError('MechanicalTurk deck changed from the source experiment')
    files = ['gandalf_deck_mix.py', 'gandalf.py', 'model_brains.py', 'gandalf_coordinate.py',
             'train_gandalf.py', 'train_gandalf_A.py', 'train_gandalf_rollout.py',
             'gandalf_resource_battle.py', 'gandalf_resource_wait.py',
             'gandalf_coordinate_battle.py', 'gandalf_entropy_battle.py',
             'gandalf_entropy.py', 'theMechanicalTurk_1.0.py']
    experiment = dict(kind='original_wizard_deck_mix_v1', source_sha256=source_hash,
                      source_completed_batches=source['completed_batches'],
                      settings=settings, decks=DECKS, opponent_deck=list(turk.DECK),
                      balancing='decision_counts_per_deck', seat_schedule='alternate_within_each_deck',
                      wait_mode='tick', critic_sampling='decision_states', discount='per_physical_tick',
                      run_seed=args.run_seed, threads=args.threads, training_temperature=1.,
                      optimizer_transfer='all_parameters',
                      code_hashes={name: file_hash(Path(__file__).parent/name) for name in files},
                      simulator_sha256=file_hash(cr_sim.__file__), cards_sha256=file_hash(CARDS_PATH),
                      torch_version=str(torch.__version__))
    return settings, experiment


def restore(args, source, settings, experiment):
    latest = args.run_dir/'latest.pt'
    if latest.exists():
        checkpoint, _ = read_checkpoint(latest)
        if checkpoint.get('config') != mixed_config() or checkpoint.get('deck_mix_experiment') != experiment:
            raise ValueError('Run directory has different source/settings/code/decks; use a new directory')
        reference = args.run_dir/'reference.pt'
        if not reference.exists() or file_hash(reference) != experiment['source_sha256']:
            raise ValueError('Frozen source is missing or changed')
    else:
        if args.run_dir.exists() and list(args.run_dir.glob('*.pt')):
            raise ValueError('Directory has snapshots but no latest.pt; use a new directory')
        checkpoint = source
    torch.manual_seed(args.run_seed)
    model = CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if any(g['lr'] != settings['learning_rate'] for g in optimizer.param_groups):
        raise ValueError('Adam learning rate differs from source training settings')
    completed = checkpoint['deck_mix_updates'] if latest.exists() else 0
    progress = checkpoint['deck_mix_progress'] if latest.exists() else empty_progress()
    return model, optimizer, completed, checkpoint['next_seed'], progress


def train(args):
    if args.source.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError('Keep source outside the mixed-deck directory')
    source, data = read_checkpoint(args.source)
    turk = load_turk()
    validate_decks(turk.DECK)
    settings, experiment = setup(args, source, hashlib.sha256(data).hexdigest(), turk)
    targets = {name: args.decision_budget//len(DECKS) for name in DECKS}
    if args.dry_run:
        _, _, completed, _, progress = restore(args, source, settings, experiment)
        print(json.dumps(experiment, indent=2))
        print(f'Validated {completed} updates; per-deck targets={targets}; '
              f"decisions={ {k:p['decisions'] for k,p in progress.items()} }; no files written")
        return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(args.run_dir/'.training.lock'), \
            (args.run_dir/'training.txt').open('a', buffering=1) as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)):
        model, optimizer, completed, next_seed, progress = restore(args, source, settings, experiment)

        def snapshot():
            return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=mixed_config(),
                        completed_batches=source['completed_batches']+completed,
                        prototype_updates=source['prototype_updates']+completed, next_seed=next_seed,
                        deck_mix_updates=completed, deck_mix_progress=progress,
                        target_decisions_per_deck=targets, deck_mix_experiment=experiment)

        if not (args.run_dir/'initial.pt').exists():
            (args.run_dir/'reference.pt').write_bytes(data)
            atomic_save(snapshot(), args.run_dir/'initial.pt')
            atomic_save(snapshot(), args.run_dir/'latest.pt')
            atomic_save(experiment, args.run_dir/'experiment.json', json_format=True)
        print(f'Mixed-deck continuation: per-deck targets={targets}; '
              f"current={ {k:p['decisions'] for k,p in progress.items()} }", flush=True)
        print('Same weights/Adam, one-tick waits, unchanged MechanicalTurk. '
              'Whole-match overshoot is reported per deck.', flush=True)
        if args.decision_budget == 0:
            print(f'Prepared frozen starting model: {args.run_dir/"initial.pt"}')
            return
        frozen = args.run_dir/f'budget_{args.decision_budget}.pt'
        if frozen.exists():
            existing, _ = read_checkpoint(frozen)
            if existing.get('deck_mix_experiment') != experiment:
                raise ValueError('Frozen budget snapshot belongs to another experiment')
            print(f'Budget snapshot already exists: {frozen}')
            return
        with patch.dict(core.SETTINGS, settings):
            while any(progress[name]['decisions'] < targets[name] for name in DECKS):
                batch_seed = args.run_seed+10_000*(source['completed_batches']+completed)
                torch.manual_seed(batch_seed)
                random.seed(batch_seed)
                names = list(DECKS)
                names = names[completed%len(names):]+names[:completed%len(names)]
                batch_target = {name: min(math.ceil(settings['target_choices']/len(DECKS)),
                                          max(0, targets[name]-progress[name]['decisions'])) for name in DECKS}
                minimum = math.ceil(settings['minimum_matches']/len(DECKS))
                rows, outcomes, seats = {k: [] for k in DECKS}, {k: Counter() for k in DECKS}, {k:[0,0] for k in DECKS}
                records, per_deck_choices = [], Counter()
                while True:
                    made_match = False
                    for name in names:
                        if not batch_target[name] or (per_deck_choices[name]>=batch_target[name] and sum(outcomes[name].values())>=minimum):
                            continue
                        # Each deck alternates independently; deck identity never determines seat.
                        seat = (progress[name]['matches']+sum(outcomes[name].values()))%2
                        match, result, stats = collect_match(
                            model, next_seed, seat, DECKS[name], turk.policy, turk.DECK, settings['gamma'])
                        next_seed += 1
                        records.extend(match)
                        rows[name].append(stats)
                        outcomes[name][result] += 1
                        seats[name][seat] += 1
                        per_deck_choices[name] += len(match)
                        made_match = True
                    if not made_match:
                        break
                print(f'\nMixed update {completed+1}: {len(records)} decisions', flush=True)
                for name in DECKS:
                    if rows[name]:
                        print(f'{name}: results={dict(outcomes[name])}; diagnostics='
                              f'{json.dumps(merge_stats(rows[name]))}', flush=True)
                returns, advantages = decision_returns(records, settings['gamma'])
                core.report_rollout_quality(records, returns, advantages)
                core.report_exploration(model, records)
                core.ppo_update(model, optimizer, records, returns, advantages)
                completed += 1
                for name in DECKS:
                    p = progress[name]
                    p['decisions'] += per_deck_choices[name]
                    p['matches'] += sum(outcomes[name].values())
                    p['seat_matches'] = [a+b for a,b in zip(p['seat_matches'], seats[name])]
                    p['results'] = dict(Counter(p['results'])+outcomes[name])
                    p['stats'] = merge_stats([p['stats'], *rows[name]])
                atomic_save(snapshot(), args.run_dir/'latest.pt')
                if completed%10 == 0:
                    atomic_save(snapshot(), args.run_dir/f'update_{completed}.pt')
                print('Saved; per-deck decisions:', {k:p['decisions'] for k,p in progress.items()}, flush=True)
        atomic_save(snapshot(), frozen)
        print(f'Training budget complete: {frozen}', flush=True)


def evaluate_one(path, args):
    checkpoint, data = read_checkpoint(path)
    if checkpoint.get('config') != mixed_config():
        raise ValueError('Evaluate initial.pt or budget snapshots from this deck-mix runner')
    model = CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint['model'])
    policy = PlacementTemperature(model, args.temperature).eval()
    turk = load_turk()
    validate_decks(turk.DECK)
    plan = dict(kind='original_wizard_evaluation_v1', checkpoint_sha256=hashlib.sha256(data).hexdigest(),
                games_per_deck=args.games, seed_start=args.seed_start, decks=DECKS,
                temperature=args.temperature, threads=args.threads,
                opponent_sha256=file_hash(Path(__file__).parent/'theMechanicalTurk_1.0.py'),
                simulator_sha256=file_hash(cr_sim.__file__),
                code_hashes={n:file_hash(Path(__file__).parent/n) for n in [
                    'gandalf_deck_mix.py','gandalf.py','model_brains.py','gandalf_coordinate.py','gandalf_resource_wait.py']})
    output = path.parent/f'evaluation_{path.stem}_{args.seed_start}_T{args.temperature:g}.json'
    with exclusive_lock(output.with_suffix('.json.lock')):
        report = dict(plan=plan, cases=[])
        if output.exists():
            report = json.loads(output.read_text())
            if ({k:v for k,v in report['plan'].items() if k!='games_per_deck'}
                    !={k:v for k,v in plan.items() if k!='games_per_deck'}):
                raise ValueError('Cached evaluation belongs to different settings; change seed-start')
            report['plan'] = plan
        done = {(c['deck'],c['seed'],c['seat']) for c in report['cases']}
        for name, deck in DECKS.items():
            for seed in range(args.seed_start,args.seed_start+args.games//2):
                for seat in (0,1):
                    if (name,seed,seat) in done:
                        continue
                    torch.manual_seed(seed*2+seat)
                    random.seed(seed*2+seat)
                    _, result, stats = collect_match(policy,seed,seat,deck,turk.policy,turk.DECK,
                                                     checkpoint['deck_mix_experiment']['settings']['gamma'])
                    report['cases'].append(dict(deck=name,seed=seed,seat=seat,result=result,stats=stats))
                    atomic_save(report,output,json_format=True)
            print(f'{path.stem}/{name}: cases complete',flush=True)
        report['summary'] = {}
        for name in DECKS:
            cases = [c for c in report['cases'] if c['deck']==name and
                     args.seed_start<=c['seed']<args.seed_start+args.games//2]
            results = Counter(c['result'] for c in cases)
            stats = merge_stats([c['stats'] for c in cases])
            report['summary'][name] = dict(results=dict(results),win_rate=results['win']/len(cases),
                seat_results=[dict(Counter(c['result'] for c in cases if c['seat']==s)) for s in (0,1)],diagnostics=stats)
            print(f"{path.stem}/{name}: {results['win']}W/{results['loss']}L/{results['draw']}D, "
                  f"win rate={results['win']/len(cases):.1%}, rejected="
                  f"{sum(v for k,v in stats['deployment_statuses'].items() if k!='ok')}",flush=True)
            if name=='wizard':
                print('Wizard exposure:',stats['card_opportunities'].get('wizard',{}),
                      'played=',stats['cards_played'].get('wizard',0),flush=True)
        atomic_save(report,output,json_format=True)
    print(f'Saved evaluation: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    trainer = commands.add_parser('train')
    trainer.add_argument('--source',type=Path,required=True)
    trainer.add_argument('--run-dir',type=Path,default=DEFAULT_DIR)
    trainer.add_argument('--decision-budget',type=int,default=25_000,help='total new decisions, split evenly by deck; 0 prepares the fork')
    trainer.add_argument('--run-seed',type=int,default=12345)
    trainer.add_argument('--dry-run',action='store_true')
    evaluator = commands.add_parser('evaluate')
    evaluator.add_argument('checkpoints',type=Path,nargs='+')
    evaluator.add_argument('--games',type=int,default=20,help='games per deck per checkpoint')
    evaluator.add_argument('--temperature',type=float,default=.25)
    evaluator.add_argument('--seed-start',type=int,default=800_000)
    for command in (trainer,evaluator):
        command.add_argument('--threads',type=int,default=1)
    args = parser.parse_args()
    if args.threads<=0:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command=='train':
        if args.decision_budget<0 or args.decision_budget%len(DECKS):
            parser.error('decision-budget must be nonnegative and divisible by the number of decks')
        train(args)
    else:
        if args.games<=0 or args.games%2 or args.seed_start<0 or not 0<args.temperature<float('inf'):
            parser.error('games must be positive/even; seed-start nonnegative; temperature finite/positive')
        for path in args.checkpoints:
            evaluate_one(path,args)


if __name__=='__main__':
    main()
