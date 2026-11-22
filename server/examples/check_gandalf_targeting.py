"""Isolated spatial-learning diagnostic; never overwrites battle checkpoints.

Optional separate diagnostic snapshots support evaluation without retraining.

Synthetic stationary enemy clusters are inserted into real observation schemas.
This is NOT a simulator battle or a Fireball damage test. Fireball is forced;
all 576 spell tiles remain available. Waiting and card selection cannot provide
a shortcut. No battle-training files or settings are changed.

Capacity mode supplies soft target labels; PPO mode uses immediate dense rewards
and a placement-only adaptation of the current PPO objective and schedule.
Held-out target centers are excluded from training.
Neither result proves that full-match attack coordination has been learned.

compare-baselines changes only the actor advantage calculation. Both branches
still train the critic and shared layers with the same value loss. It tests the
critic's role in actor feedback, not whether shared critic gradients interfere.
compare-critic-loss uses batch-mean advantages in both runs and changes only the
value-loss coefficient, testing the effect of critic training on shared updates.
compare-gradients compares sampled actor feedback with the old-policy-weighted
PPO surrogate summed over all tiles. Sampled normalization, critic targets, and
KL stopping remain unchanged; only the actor action expectation is enumerated.
The all-tiles branch has extra oracle reward information, not an equal reward-
query budget, and is only meaningful for this synthetic diagnostic.
compare-heads tests the original tile head against a shared coordinate scorer
under sampled PPO. Common encoder/trunk weights and state budgets match; head
parameters and initial action distributions necessarily differ.
"""

import argparse
import copy
import random
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import cr_sim
import torch
from torch.distributions import Categorical

from gandalf import (
    CARD_NAMES, DECK, NUM_TILES, action_masks, decode_tile, encode_values,
)
from model_brains import ActorCritic, batch_inputs, evaluate_actions
from gandalf_coordinate import CoordinateTileHead
from train_gandalf import (
    SETTINGS, calculate_returns_and_advantages,
)


TILES = torch.tensor([decode_tile(i) for i in range(NUM_TILES)])
TRAIN_CENTERS = [(x + .5, y + .5) for y in range(9, 23)
                 for x in range(2, 16) if (x + y) % 4 != 0]
TEST_CENTERS = [(x + .5, y + .5) for y in range(9, 23)
                for x in range(2, 16) if (x + y) % 4 == 0]


def make_model(placement_head='original'):
    # Build the original model first so all common weights have the same seeded
    # initialization. Replace only the tile head in this in-memory prototype.
    model = ActorCritic(len(CARD_NAMES))
    if placement_head == 'coordinates':
        model.tile_head = CoordinateTileHead()
    elif placement_head != 'original':
        raise ValueError(f'Unknown placement head: {placement_head}')
    return model


@lru_cache(maxsize=1)
def template():
    battle = cr_sim.Battle()
    battle.reset(seed=42, deck0=DECK, deck1=DECK)
    obs = copy.deepcopy(battle.view(0).observe())
    obs.update(hand=['fireball', 'knight', 'archers', 'giant'],
               hand_cost=[4, 3, 3, 5], legal=[True] * 4,
               next_card='musketeer', elixir=10.0)
    obs['entities'] = [e for e in obs['entities'] if e[7]]
    _, tiles = action_masks(obs)
    assert all(tiles[0]), 'Fireball must retain all 576 targeting options'
    # Only Fireball can be selected. Unused branches use empty masks just as
    # unaffordable cards do in normal play. This mask is identical for all cases.
    tiles = [tiles[0]] + [[False] * NUM_TILES for _ in range(3)]
    return obs, [False, True, False, False, False], tiles


def fixture(seed, centers):
    """Use a private RNG so fixture generation cannot alter policy sampling."""
    rng = random.Random(seed)
    base, slots, tiles = template()
    obs = copy.deepcopy(base)
    x, y = rng.choice(centers)
    x += rng.uniform(-.2, .2)
    y += rng.uniform(-.2, .2)
    # Three living enemies, plus an unrelated friendly unit as a distractor.
    for i, (dx, dy) in enumerate([(-.35, -.2), (.35, -.2), (0, .4)]):
        obs['entities'].append(
            [1, 'archers', x + dx, y + dy, 300, 300,
             False, False, False, 100 + i, 0., 0.])
    obs['entities'].append(
        [0, 'knight', rng.uniform(2, 16), rng.uniform(9, 23),
         1000, 1000, False, False, False, 200, 0., 0.])
    return encode_values(obs), slots, tiles, (x, y)


def make_batch(seed, count, centers):
    fixtures = [fixture(seed + i, centers) for i in range(count)]
    encoded, slots, tiles, targets = zip(*fixtures)
    return (list(encoded), torch.tensor(slots, dtype=torch.bool),
            torch.tensor(tiles, dtype=torch.bool), torch.tensor(targets))


@torch.no_grad()
def evaluate(model, batch):
    encoded, _, _, targets = batch
    _, logits, _ = model(batch_inputs(encoded))
    logits = logits[:, 0]
    probabilities = logits.softmax(-1)
    d2 = (TILES[None] - targets[:, None]).square().sum(-1)
    chosen = logits.argmax(-1)
    rows = torch.arange(len(encoded))
    hit = d2 <= 2.5 ** 2  # Diagnostic distance threshold, not simulator damage.
    # A successful conditional policy should fail when its targets are shuffled.
    shuffled = (TILES[chosen] - targets.roll(1, 0)).square().sum(-1)
    result = {
        'greedy_hit': hit[rows, chosen].float().mean().item(),
        'sampled_hit': (probabilities * hit).sum(-1).mean().item(),
        'shuffled_hit': (shuffled <= 2.5 ** 2).float().mean().item(),
        'mean_distance': d2[rows, chosen].sqrt().mean().item(),
        'tile_entropy': Categorical(logits=logits).entropy().mean().item(),
        'top_tile_frequency': chosen.bincount(minlength=NUM_TILES).max().item()
                              / len(encoded),
    }
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())
    return result


def show(label, metrics):
    print(f"{label}: greedy hit={metrics['greedy_hit']:.1%}, "
          f"sampled hit={metrics['sampled_hit']:.1%}, "
          f"shuffled-target hit={metrics['shuffled_hit']:.1%}, "
          f"mean distance={metrics['mean_distance']:.2f}, "
          f"tile entropy={metrics['tile_entropy']:.3f}, "
          f"top-tile frequency={metrics['top_tile_frequency']:.1%}", flush=True)


def temperature_metrics(logits, targets, temperature):
    """Exact expected sampling metrics; no random draws or training changes."""
    d2 = (TILES[None] - targets[:, None]).square().sum(-1)
    shuffled_d2 = (TILES[None] - targets.roll(1, 0)[:, None]).square().sum(-1)
    if temperature == 0:  # Greedy reference, not a sampling temperature.
        probabilities = torch.nn.functional.one_hot(
            logits.argmax(-1), NUM_TILES).to(logits.dtype)
        entropy = 0.
    else:
        if not 0 < temperature < float('inf'):
            raise ValueError('Temperature must be finite and positive')
        distribution = Categorical(logits=logits / temperature)
        probabilities = distribution.probs
        entropy = distribution.entropy().mean().item()
    return dict(
        hit=(probabilities * (d2 <= 2.5 ** 2)).sum(-1).mean().item(),
        shuffled_hit=(probabilities * (shuffled_d2 <= 2.5 ** 2)).sum(-1).mean().item(),
        distance=(probabilities * d2.sqrt()).sum(-1).mean().item(),
        entropy=entropy,
    )


@torch.no_grad()
def show_temperatures(model, batch, temperatures):
    encoded, _, _, targets = batch
    _, logits, _ = model(batch_inputs(encoded))
    print('\nHELD-OUT TEMPERATURE SWEEP (evaluation only; same weights/targets):')
    results = {}
    for temperature in [*temperatures, 0.]:
        metrics = temperature_metrics(logits[:, 0], targets, temperature)
        label = f'T={temperature:g}' if temperature else 'Greedy'
        results[label] = metrics
        print(f"{label}: hit={metrics['hit']:.1%}, "
              f"shuffled-target hit={metrics['shuffled_hit']:.1%}, "
              f"expected distance={metrics['distance']:.2f}, "
              f"entropy={metrics['entropy']:.3f}", flush=True)
    print('Sampling hit rates are exact probability-weighted expectations, '
          'not additional sampled games. Training always uses temperature 1.')
    return results


def evaluate_diagnostic(path, temperatures):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('kind') != 'gandalf_targeting_diagnostic_v1'
            or saved.get('card_names') != CARD_NAMES or saved.get('deck') != DECK):
        raise ValueError('Not a compatible targeting diagnostic checkpoint')
    model = make_model(saved['placement_head'])
    model.load_state_dict(saved['model'])
    model.eval()
    held_out = make_batch(900_000 + saved['seed'], 128, TEST_CENTERS)
    print(f"Evaluating diagnostic {path}: seed={saved['seed']}, "
          f"steps={saved['steps']}, head={saved['placement_head']}")
    show('Saved model / held out', evaluate(model, held_out))
    show_temperatures(model, held_out, temperatures)


def actor_advantages(records, returns, critic_advantages, baseline):
    """Detached one-step feedback; downstream normalization stays identical."""
    if baseline == 'critic':
        return critic_advantages.detach()
    if baseline != 'batch-mean':
        raise ValueError(f'Unknown actor baseline: {baseline}')
    assert all(r['done'] for r in records), 'Batch-mean test requires one-step tasks'
    rewards = torch.tensor([r['reward'] for r in records], dtype=returns.dtype)
    assert torch.allclose(returns, rewards, atol=1e-6)
    # This removes the learned value estimate only from the actor's feedback.
    # Subtracting a constant is redundant with later advantage centering, but
    # spells out the simple baseline being tested. Critic targets stay unchanged.
    return (rewards - rewards.mean()).detach()


def actor_surrogate(log_probs, old_log_probs, advantages, clip, weights=None):
    """Sample average, or exact old-policy expectation across the action axis."""
    ratio = (log_probs - old_log_probs).exp()
    surrogate = torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    if weights is None:
        return -surrogate.mean()
    return -(weights * surrogate).sum(-1).mean()


def placement_update(model, optimizer, records, returns, advantages,
                     all_rewards=None, baseline='critic'):
    """Match the battle PPO loss/schedule, but include forced-card tile choices.

    The battle updater selects actor ticks using >1 legal slot, which deliberately
    excludes these fixtures. Keep this adaptation local so battle training and
    checkpoint code hashes remain unchanged. Slot log-probability/entropy is zero;
    the joint action probability is exactly the selected tile probability.
    """
    encoded = batch_inputs([r['observations'] for r in records])
    slots = torch.tensor([r['slot_mask'] for r in records], dtype=torch.bool)
    tiles = torch.tensor([r['tile_masks'] for r in records], dtype=torch.bool)
    actions = torch.ones(len(records), dtype=torch.long)
    chosen = torch.tensor([r['tile_id'] for r in records])
    old = torch.tensor([r['log_probability'] for r in records])
    center = advantages.mean()
    scale = advantages.std(unbiased=False) + 1e-8
    advantages = (advantages - center) / scale
    if all_rewards is not None:
        assert all_rewards.shape == (len(records), NUM_TILES)
        assert torch.isfinite(all_rewards).all()
        with torch.no_grad():
            _, tile_logits, _ = model(encoded)
            old_tile_logs = tile_logits[:, 0].log_softmax(-1)
            old_tile_probs = old_tile_logs.exp()
            if baseline == 'critic':
                offsets = torch.tensor([r['value_estimate'] for r in records])[:, None]
            elif baseline == 'batch-mean':
                offsets = torch.tensor([r['reward'] for r in records]).mean()
            else:
                raise ValueError(f'Unknown actor baseline: {baseline}')
            # Use the same sampled centering/scaling as the sampled branch.
            # This isolates action summation, not critic or normalization noise.
            all_advantages = ((all_rewards - offsets - center) / scale).detach()

    def policy_kl():
        with torch.no_grad():
            log_probs, _, _ = evaluate_actions(
                model, encoded, actions, chosen, slots, tiles)
            log_ratio = log_probs - old
            return (log_ratio.exp() - 1 - log_ratio).mean().item()

    assert policy_kl() < 1e-5, 'Collection/update probability mismatch'
    diagnostics = []
    for _ in range(SETTINGS['ppo_epochs']):
        order = torch.randperm(len(records))
        for start in range(0, len(records), SETTINGS['actor_batch_size']):
            indices = order[start:start + SETTINGS['actor_batch_size']]
            log_probs, entropy, _ = evaluate_actions(
                model, {k: v[indices] for k, v in encoded.items()},
                actions[indices], chosen[indices], slots[indices], tiles[indices])
            ratio = (log_probs - old[indices]).exp()
            adv = advantages[indices]
            clip = SETTINGS['clip_range']
            if all_rewards is None:
                policy_loss = actor_surrogate(
                    log_probs, old[indices], adv, clip)
            else:
                _, tile_logits, _ = model({k: v[indices] for k, v in encoded.items()})
                policy_loss = actor_surrogate(
                    tile_logits[:, 0].log_softmax(-1), old_tile_logs[indices],
                    all_advantages[indices], clip, old_tile_probs[indices])
            critic_indices = torch.randint(
                len(records), (SETTINGS['critic_batch_size'],))
            _, _, values = model({k: v[critic_indices] for k, v in encoded.items()})
            value_loss = (values - returns[critic_indices]).square().mean()
            loss = (policy_loss + SETTINGS['value_coefficient'] * value_loss
                    - SETTINGS['entropy_coefficient'] * entropy.mean())
            loss = loss * (len(indices) / SETTINGS['actor_batch_size'])
            assert torch.isfinite(loss), 'Non-finite placement PPO loss'
            optimizer.zero_grad()
            loss.backward()
            tile_grad = sum(p.grad.square().sum().item()
                            for p in model.tile_head.parameters()
                            if p.grad is not None) ** .5
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), SETTINGS['max_grad_norm'],
                error_if_nonfinite=True).item()
            optimizer.step()
            kl = policy_kl()
            diagnostics.append(dict(kl=kl, grad=grad, tile_grad=tile_grad,
                                    clip_fraction=((ratio - 1).abs() > clip)
                                    .float().mean().item()))
            assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics[-1].values())
            if kl > SETTINGS['target_kl']:
                return diagnostics
    return diagnostics


def run(mode, args, entropy_coefficient=None, baseline='critic', value_coefficient=None,
        gradient_estimator='sampled', placement_head='original'):
    torch.manual_seed(args.seed)
    model = make_model(placement_head)
    initial_tile_weights = {k: p.detach().clone()
                            for k, p in model.tile_head.named_parameters()}
    initial_value_weights = {k: v.detach().clone()
                             for k, v in model.value_head.state_dict().items()}
    learning_rate = getattr(args, 'learning_rate', None)
    if learning_rate is None:
        learning_rate = .001 if mode == 'capacity' else SETTINGS['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    held_out = make_batch(900_000 + args.seed, 128, TEST_CENTERS)
    train_probe = make_batch(800_000 + args.seed, 128, TRAIN_CENTERS)
    coefficient = SETTINGS['entropy_coefficient'] if entropy_coefficient is None else entropy_coefficient
    value_coefficient = SETTINGS['value_coefficient'] if value_coefficient is None else value_coefficient
    print(f'\n{mode.upper()}: fresh weights, seed={args.seed}, '
          f'{args.steps} updates, learning rate={learning_rate}, '
          f'entropy coefficient={coefficient}, '
          f'actor baseline={baseline}, value-loss coefficient={value_coefficient}, '
          f'actor gradient={gradient_estimator}, placement head={placement_head}, '
          f'placement only, {args.steps * args.batch_size} one-step samples', flush=True)
    print(f'Tile-head trainable parameters: '
          f'{sum(p.numel() for p in model.tile_head.parameters())}', flush=True)
    if gradient_estimator == 'all-tiles':
        print('Oracle diagnostic: actor uses rewards for all 576 tiles per state; '
              'normalization, critic targets and KL stopping still use samples.', flush=True)
    before = evaluate(model, held_out)
    show('Before / held out', before)
    window_diagnostics = []
    for step in range(args.steps):
        # Paired runs share initialization, fixtures, and per-update random seeds.
        # Policies may differ; common uniform draws keep sampling comparable.
        torch.manual_seed(args.seed + 1_000_000 + step)
        encoded, slots, tiles, targets = make_batch(
            args.seed + step * args.batch_size, args.batch_size, TRAIN_CENTERS)
        inputs = batch_inputs(encoded)
        d2 = (TILES[None] - targets[:, None]).square().sum(-1)
        if mode == 'capacity':
            _, logits, _ = model(inputs)
            # Smooth labels give nearby tiles credit, including withheld centers.
            labels = (-d2 / (2 * 1.5 ** 2)).softmax(-1)
            loss = -(labels * logits[:, 0].log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5,
                                          error_if_nonfinite=True)
            optimizer.step()
        else:
            with torch.no_grad():
                _, logits, _ = model(inputs)
                cdf = logits[:, 0].softmax(-1).cumsum(-1)
                chosen = torch.searchsorted(
                    cdf.contiguous(), torch.rand(args.batch_size, 1)
                ).squeeze(-1).clamp_max(NUM_TILES - 1)
                actions = torch.ones(args.batch_size, dtype=torch.long)
                logs, _, values = evaluate_actions(
                    model, inputs, actions, chosen, slots, tiles)
                # Deliberately easy, dense one-step feedback, not match reward.
                reward_table = (-d2 / (2 * 4.0 ** 2)).exp()
                rewards = reward_table[torch.arange(args.batch_size), chosen]
            records = [dict(
                observations=encoded[i], slot_action=actions[i].item(),
                tile_id=chosen[i].item() if actions[i] == 1 else None,
                slot_mask=slots[i].tolist(),
                tile_masks=tiles[i].tolist(), log_probability=logs[i].item(),
                value_estimate=values[i].item(), reward=rewards[i].item(),
                done=True) for i in range(args.batch_size)]
            returns, advantages = calculate_returns_and_advantages(
                records, gamma=SETTINGS['gamma'], gae_lambda=1.)
            advantages = actor_advantages(records, returns, advantages, baseline)
            # Keep critic sampling/computation identical even when its weighted
            # loss is zero. This avoids changing the random-number sequence.
            with patch.dict(SETTINGS, {'entropy_coefficient': coefficient,
                                      'value_coefficient': value_coefficient}):
                if gradient_estimator == 'all-tiles':
                    diagnostics = placement_update(
                        model, optimizer, records, returns, advantages,
                        all_rewards=reward_table, baseline=baseline)
                else:
                    diagnostics = placement_update(
                        model, optimizer, records, returns, advantages)
                window_diagnostics.extend(diagnostics)
        if (step + 1) % args.report_every == 0 or step + 1 == args.steps:
            show(f'Update {step + 1} / train', evaluate(model, train_probe))
            show(f'Update {step + 1} / held out', evaluate(model, held_out))
            if window_diagnostics:
                n = len(window_diagnostics)
                print('PPO diagnostics / reporting window: '
                      f"last KL={window_diagnostics[-1]['kl']:.6f}, "
                      f"max KL={max(d['kl'] for d in window_diagnostics):.6f}, "
                      f"mean pre-clip gradient norm={sum(d['grad'] for d in window_diagnostics)/n:.4f}, "
                      f"mean tile-head gradient norm={sum(d['tile_grad'] for d in window_diagnostics)/n:.4f}, "
                      f"mean clip fraction={sum(d['clip_fraction'] for d in window_diagnostics)/n:.3f}",
                      flush=True)
                window_diagnostics.clear()
    final = evaluate(model, held_out)
    assert any(not torch.equal(initial_tile_weights[k], p)
               for k, p in model.tile_head.named_parameters()), (
        'Tile weights did not change: the diagnostic did not train placement')
    if mode == 'ppo' and value_coefficient == 0:
        assert all(torch.equal(initial_value_weights[k], v)
                   for k, v in model.value_head.state_dict().items()), (
            'Value-head weights changed with critic loss disabled')
    print(f"Held-out greedy improvement: "
          f"{100 * (final['greedy_hit'] - before['greedy_hit']):+.1f} points")
    print('Interpretation: look for improved held-out hit rates well above the '
          'shuffled-target control; entropy reduction alone is insufficient.')
    print('Fireball is forced; hit rates measure placement only.')
    show_temperatures(model, held_out, getattr(args, 'eval_temperatures', [1., .5, .25]))
    diagnostic_path = getattr(args, 'save_diagnostic', None)
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        saved = dict(kind='gandalf_targeting_diagnostic_v1',
                     model=model.state_dict(), card_names=CARD_NAMES, deck=DECK,
                     placement_head=placement_head, seed=args.seed, steps=args.steps,
                     settings={**SETTINGS, 'learning_rate': learning_rate,
                               'entropy_coefficient': coefficient,
                               'value_coefficient': value_coefficient},
                     baseline=baseline, gradient_estimator=gradient_estimator)
        # Never overwrite any existing model, including diagnostic snapshots.
        with diagnostic_path.open('xb') as destination:
            torch.save(saved, destination)
        print(f'Saved separate diagnostic weights: {diagnostic_path}')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=[
        'capacity', 'ppo', 'both', 'compare', 'compare-baselines',
        'compare-critic-loss', 'compare-gradients', 'compare-heads'],
                        default='compare')
    parser.add_argument('--entropy-coefficient', type=float, default=None,
                        help='coefficient for a single run or baseline comparison '
                             '(baseline/critic-loss comparisons default to 0.05)')
    parser.add_argument('--baseline', choices=['critic', 'batch-mean'],
                        default='critic', help='actor baseline for single PPO runs')
    parser.add_argument('--value-coefficient', type=float, default=None,
                        help='value-loss coefficient for a single run; 0 disables '
                             'critic gradients (use batch-mean advantages)')
    parser.add_argument('--gradient-estimator', choices=['sampled', 'all-tiles'],
                        default='sampled', help='actor estimator for a single PPO run')
    parser.add_argument('--placement-head', choices=['original', 'coordinates'],
                        default='original', help='placement head for a single run')
    parser.add_argument('--entropy-coefficients', type=float, nargs=2,
                        default=[SETTINGS['entropy_coefficient'], .05],
                        help='paired coefficients for compare mode')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Adam learning rate for this diagnostic only; '
                             'defaults to existing settings (capacity: 0.001)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=31415)
    parser.add_argument('--report-every', type=int, default=50)
    parser.add_argument('--eval-temperatures', type=float, nargs='+', default=[1., .5, .25],
                        help='evaluation-only temperatures; greedy is always included')
    parser.add_argument('--save-diagnostic', type=Path,
                        help='save a new diagnostic snapshot; never overwrites a file')
    parser.add_argument('--evaluate-diagnostic', type=Path,
                        help='evaluate saved diagnostic weights without any training')
    args = parser.parse_args()
    if not all(0 < t < float('inf') for t in args.eval_temperatures):
        parser.error('evaluation temperatures must be finite and positive')
    if args.evaluate_diagnostic is not None:
        if args.save_diagnostic is not None:
            parser.error('choose evaluation or saving, not both')
        torch.set_num_threads(1)
        evaluate_diagnostic(args.evaluate_diagnostic, args.eval_temperatures)
        return
    if args.save_diagnostic is not None:
        if args.mode not in ['ppo', 'capacity']:
            parser.error('saving diagnostic weights requires a single ppo/capacity run')
        if args.save_diagnostic.exists():
            parser.error('diagnostic destination already exists; refusing to overwrite')
    if min(args.steps, args.batch_size, args.report_every) <= 0:
        parser.error('steps, batch-size and report-every must be positive')
    if args.learning_rate is not None and not (
            0 < args.learning_rate < float('inf')):
        parser.error('learning rate must be finite and positive')
    if not all(0 <= c < float('inf') for c in args.entropy_coefficients):
        parser.error('entropy coefficients must be finite and nonnegative')
    if args.entropy_coefficient is not None and not (
            0 <= args.entropy_coefficient < float('inf')):
        parser.error('entropy coefficient must be finite and nonnegative')
    if args.mode == 'compare' and args.entropy_coefficient is not None:
        parser.error('use --entropy-coefficients for entropy comparison mode')
    if args.mode == 'compare-heads' and args.gradient_estimator != 'sampled':
        parser.error('head comparison uses sampled PPO, not oracle action rewards')
    if args.value_coefficient is not None:
        if not 0 <= args.value_coefficient < float('inf'):
            parser.error('value coefficient must be finite and nonnegative')
        if args.mode in ['compare', 'compare-baselines', 'compare-critic-loss']:
            parser.error('--value-coefficient is for single runs, not paired comparisons')
        if args.value_coefficient == 0 and args.baseline != 'batch-mean':
            parser.error('use --baseline batch-mean when disabling critic loss')
    torch.set_num_threads(1)
    # Override only inside this process; real trainer settings stay unchanged.
    with patch.dict(SETTINGS, {'actor_batch_size': args.batch_size}):
        if args.mode == 'compare-heads':
            coefficient = .05 if args.entropy_coefficient is None else args.entropy_coefficient
            results = [(head, run('ppo', args, coefficient, args.baseline,
                                  args.value_coefficient, 'sampled', head))
                       for head in ['original', 'coordinates']]
            print('\nPAIRED PLACEMENT HEAD COMPARISON '
                  '(sampled PPO, same common initialization and state budget):')
            for head, metrics in results:
                show(f'Placement head {head}', metrics)
            print('Head parameter counts and initial action distributions differ; '
                  'this tests the architecture change, not parameter-matched capacity.')
        elif args.mode == 'compare-gradients':
            coefficient = .05 if args.entropy_coefficient is None else args.entropy_coefficient
            results = [(estimator, run('ppo', args, coefficient, args.baseline,
                                       args.value_coefficient, estimator))
                       for estimator in ['sampled', 'all-tiles']]
            print('\nPAIRED ACTOR GRADIENT COMPARISON '
                  '(same state budget, extra oracle action rewards in all-tiles):')
            for estimator, metrics in results:
                show(f'Actor gradient {estimator}', metrics)
        elif args.mode == 'compare-critic-loss':
            coefficient = .05 if args.entropy_coefficient is None else args.entropy_coefficient
            control = SETTINGS['value_coefficient']
            if control <= 0:
                parser.error('critic-loss comparison requires a positive control value coefficient')
            results = [(v, run('ppo', args, coefficient, 'batch-mean', v))
                       for v in [control, 0.]]
            print('\nPAIRED CRITIC LOSS COMPARISON '
                  '(batch-mean advantages in both, same entropy and sample budget):')
            for value_coefficient, metrics in results:
                show(f'Value-loss coefficient {value_coefficient}', metrics)
            print('Only the weighted critic loss changed. Zero disables critic '
                  'gradients; actor/entropy updates still train shared layers. '
                  'The same global gradient clipping applies in both branches.')
        elif args.mode == 'compare-baselines':
            coefficient = .05 if args.entropy_coefficient is None else args.entropy_coefficient
            results = [(baseline, run('ppo', args, coefficient, baseline))
                       for baseline in ['critic', 'batch-mean']]
            print('\nPAIRED ACTOR BASELINE COMPARISON '
                  '(same sample budget, entropy coefficient, and critic loss):')
            for baseline, metrics in results:
                show(f'Actor baseline {baseline}', metrics)
            print('Only actor feedback changed. This comparison does not '
                  'remove critic gradients from shared layers.')
        elif args.mode == 'compare':
            results = [(c, run('ppo', args, c)) for c in args.entropy_coefficients]
            print('\nPAIRED ENTROPY COMPARISON (same sample budget):')
            for coefficient, metrics in results:
                show(f'Entropy coefficient {coefficient}', metrics)
        else:
            for mode in (['capacity', 'ppo'] if args.mode == 'both' else [args.mode]):
                run(mode, args, args.entropy_coefficient, args.baseline,
                    args.value_coefficient, args.gradient_estimator,
                    args.placement_head)
    print('\nFinished. Battle files and existing checkpoints unchanged.')


if __name__ == '__main__':
    main()
