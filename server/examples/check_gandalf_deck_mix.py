"""Real-simulator integration checks; scratch checkpoints are removed afterward."""

import argparse
import contextlib
import io
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cr_sim
import torch

import gandalf_deck_mix as mix
from gandalf import CARD_INDEX, CARD_NAMES, DECK, action_masks, encode_values
from gandalf_coordinate import CoordinateActorCritic
from gandalf_resource_wait import collect_resource_match, decision_returns


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=mix.ROOT/'checkpoints/gandalf/resource_tick_v1/budget_25000.pt')
    args = parser.parse_args()
    torch.set_num_threads(1)
    source, _ = mix.read_checkpoint(args.source)
    source_hash = mix.file_hash(args.source)
    turk = mix.load_turk()
    turk_path = Path(mix.__file__).parent/'theMechanicalTurk_1.0.py'
    turk_hash = mix.file_hash(turk_path)
    original_deck = list(DECK)
    mix.validate_decks(turk.DECK)
    model = CoordinateActorCritic(len(CARD_NAMES))
    model.load_state_dict(source['model'])
    gamma = source['resource_experiment']['settings']['gamma']

    # Verify the card identity/cost and legal placement mask in both seats.
    for seat in (0, 1):
        seen = False
        for seed in range(30):
            battle = cr_sim.Battle()
            decks = [turk.DECK, turk.DECK]
            decks[seat] = mix.DECKS['wizard']
            battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])
            obs = battle.view(seat).observe()
            if 'wizard' not in obs['hand']:
                continue
            index = obs['hand'].index('wizard')
            encoded = encode_values(obs)
            assert encoded['hand_ids'][index] == CARD_INDEX['wizard']
            assert encoded['hand_features'][index][0] == .5
            obs['elixir'], obs['legal'] = 10., [True]*4
            slots, tiles = action_masks(obs)
            assert slots[index+1] and any(tiles[index])
            seen = True
            break
        assert seen
    print('Wizard identity, cost, and placement masks passed in both seats.')

    for seat in (0, 1):
        torch.manual_seed(987+seat)
        random.seed(987+seat)
        expected, result, stats = collect_resource_match(
            model, 900000+seat, turk.policy, turk.DECK, seat=seat, gamma=gamma)
        torch.manual_seed(987+seat)
        random.seed(987+seat)
        actual, actual_result, actual_stats = mix.collect_match(
            model, 900000+seat, seat, mix.DECKS['original'], turk.policy, turk.DECK, gamma)
        assert expected == actual and result == actual_result
        assert stats['deployment_statuses'] == actual_stats['deployment_statuses']
        assert all(k == 'ok' for k in actual_stats['deployment_statuses'])
        returns, _ = decision_returns(actual, gamma)
        terminal_tick = actual[-1]['start_tick']+actual[-1]['duration_ticks']
        outcome = {'win': 1., 'loss': -1., 'draw': 0.}[result]
        expected_returns = torch.tensor([
            outcome*gamma**(terminal_tick-r['start_tick']-1) for r in actual])
        assert torch.allclose(returns, expected_returns, atol=1e-6)
    print('Original-deck trajectories match the existing collector; physical-time returns passed.')

    # Small real training run covers both decks, both seats, PPO, and serialization.
    with tempfile.TemporaryDirectory(prefix='gandalf_deck_mix_') as temporary:
        run = SimpleNamespace(source=args.source.resolve(), run_dir=Path(temporary),
                              decision_budget=2, run_seed=12345, threads=1, dry_run=False)
        with contextlib.redirect_stdout(io.StringIO()):
            mix.train(run)
        initial, _ = mix.read_checkpoint(run.run_dir/'initial.pt')
        latest, latest_bytes = mix.read_checkpoint(run.run_dir/'latest.pt')
        assert equal_state(initial['model'], source['model'])
        assert equal_state(initial['optimizer'], source['optimizer'])
        assert not equal_state(latest['model'], initial['model'])
        for name, progress in latest['deck_mix_progress'].items():
            assert progress['decisions'] > 0 and progress['matches'] >= 4
            assert abs(progress['seat_matches'][0]-progress['seat_matches'][1]) <= 1
            assert all(k == 'ok' for k in progress['stats']['deployment_statuses'])
        with contextlib.redirect_stdout(io.StringIO()):
            mix.train(run)
        assert (run.run_dir/'latest.pt').read_bytes() == latest_bytes
        turk_module = mix.load_turk()
        settings, experiment = mix.setup(run, source, source_hash, turk_module)
        restored, optimizer, updates, _, progress = mix.restore(run, source, settings, experiment)
        assert equal_state(restored.state_dict(), latest['model'])
        assert equal_state(optimizer.state_dict(), latest['optimizer'])
        assert updates == latest['deck_mix_updates'] and progress == latest['deck_mix_progress']
        eval_args = SimpleNamespace(games=2, seed_start=910000, temperature=.25, threads=1)
        frozen = run.run_dir/'budget_2.pt'
        with contextlib.redirect_stdout(io.StringIO()):
            mix.evaluate_one(frozen, eval_args)
        report = frozen.parent/'evaluation_budget_2_910000_T0.25.json'
        cached = report.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            mix.evaluate_one(frozen, eval_args)
        assert report.read_bytes() == cached
        evaluation = json.loads(cached)
        assert len(evaluation['cases']) == 4
        assert set(evaluation['summary']) == set(mix.DECKS)
    assert DECK == original_deck
    assert mix.file_hash(args.source) == source_hash and mix.file_hash(turk_path) == turk_hash
    print('Mixed-deck PPO, exact weight/Adam transfer, resume, and cached evaluation passed.')
    print('Source checkpoint, MechanicalTurk, and global DECK are unchanged.')


if __name__ == '__main__':
    main()
