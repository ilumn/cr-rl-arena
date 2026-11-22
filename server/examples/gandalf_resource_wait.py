"""Bounded voluntary waiting and decision-boundary MC transitions.

The policy still chooses among wait and four hand cards. Reserve mode changes
only the meaning of a selected wait; it never overrides a selected deployment.
Forced ticks produce no actor/critic examples. Each decision records its physical
duration and discounted interval reward, so MC returns keep per-tick discounting.
"""

from collections import Counter

import cr_sim
import torch

from gandalf import DECK, action_masks, decode_action, encode_values
from model_brains import select_action


class ResourceHold:
    def __init__(self, max_ticks, threat_line=17.):
        if max_ticks < 1 or not 0 <= threat_line <= 32:
            raise ValueError('Invalid hold limit or threat line')
        self.max_ticks = max_ticks
        self.threat_line = threat_line
        self.active = False
        self.counts = Counter()

    def danger(self, obs):
        return any(e[0] == 1 and e[4] > 0 and not e[7] and e[3] < self.threat_line
                   for e in obs['entities'])

    def start(self, obs):
        self.counts['selected'] += 1
        if self.danger(obs):
            self.counts['denied_existing_threat'] += 1
            return False
        if not any(obs['legal']):
            self.counts['denied_no_affordable_card'] += 1
            return False
        elixir = obs['elixir']
        higher = [float(cost) for cost in obs['hand_cost'] if cost > elixir + 1e-9]
        self.target = min(10., min(higher) if higher else elixir + 1.)
        if self.target <= elixir:
            self.counts['denied_at_cap'] += 1
            return False
        self.started_at = obs['tick']
        self.start_legal = list(obs['legal'])
        self.towers = {e[9]: e[4] for e in obs['entities'] if e[0] == 0 and e[7] and e[4] > 0}
        self.active = True
        self.counts['started'] += 1
        return True

    def continue_holding(self, obs):
        if not self.active:
            return False
        current = {e[9]: e[4] for e in obs['entities'] if e[0] == 0 and e[7]}
        reason = None
        if self.danger(obs):
            reason = 'incoming_threat'
        elif any(current.get(uid, 0) < hp for uid, hp in self.towers.items()):
            reason = 'tower_damage'
        elif any(now and not old for now, old in zip(obs['legal'], self.start_legal)):
            reason = 'card_unlocked'
        elif obs['elixir'] >= self.target:
            reason = 'elixir_target'
        elif obs['tick'] - self.started_at >= self.max_ticks:
            reason = 'time_limit'
        if reason:
            self.active = False
            self.counts[f'released_{reason}'] += 1
            return False
        return True


def decision_returns(records, gamma):
    """R_interval + gamma**duration * G_next, without crossing episode ends.

    Terminal interval reward was already discounted by gamma**(duration-1).
    Thus a win T ticks after a decision has the same return as the original
    full-tick MC calculation: gamma**(T-1).
    """
    if not records or not records[-1]['done']:
        raise ValueError('Expected nonempty, complete episodes')
    returns = [0.] * len(records)
    running = 0.
    for i in reversed(range(len(records))):
        record = records[i]
        duration = record['duration_ticks']
        if not isinstance(duration, int) or duration < 1:
            raise ValueError('Each decision must span at least one physical tick')
        if record['done']:
            running = 0.
        running = record['reward'] + (0. if record['done'] else gamma ** duration * running)
        returns[i] = running
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - torch.tensor([r['value_estimate'] for r in records])
    assert torch.isfinite(returns).all() and torch.isfinite(advantages).all()
    return returns, advantages


@torch.no_grad()
def collect_resource_match(model, seed, opponent_fn, opponent_deck, seat=0,
                           wait_mode='tick', max_hold_ticks=60, threat_line=17., gamma=.9999):
    if wait_mode not in ('tick', 'reserve'):
        raise ValueError('Unknown wait mode')
    battle = cr_sim.Battle()
    decks = [None, None]
    decks[seat], decks[1-seat] = DECK, opponent_deck
    battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])
    view, other = battle.view(seat), battle.view(1-seat)
    hold = ResourceHold(max_hold_ticks, threat_line)
    records, cards, deployments, counts = [], Counter(), Counter(), Counter()
    elixir_sum = 0.
    while not battle.done():
        obs = view.observe()
        action = None
        selected = None
        if hold.continue_holding(obs):
            counts['forced_hold_ticks'] += 1
        elif not any(obs['legal']):
            counts['forced_unaffordable_ticks'] += 1
        else:
            slots, tiles = action_masks(obs)
            if sum(slots) > 1:
                encoded = encode_values(obs)
                selected, tile, log_prob, value = select_action(model, encoded, slots, tiles)
                action = decode_action(selected, tile)
                record = dict(observations=encoded, slot_action=selected, tile_id=tile,
                              slot_mask=slots, tile_masks=tiles, log_probability=log_prob,
                              value_estimate=value, reward=0., done=False,
                              start_tick=obs['tick'], duration_ticks=0)
                records.append(record)
                elixir_sum += obs['elixir']
                counts['decisions'] += 1
                counts['multi_card_decisions'] += sum(obs['legal']) >= 2
                counts['voluntary_waits'] += selected == 0
                if selected:
                    cards[obs['hand'][selected-1]] += 1
                elif wait_mode == 'reserve' and hold.start(obs):
                    record['hold_target_elixir'] = hold.target
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
            deployments[after['last_action']] += 1
        if records:
            latest = records[-1]
            latest['duration_ticks'] = int(after['tick'] - latest['start_tick'])
            latest['done'] = battle.done()
            if latest['done']:
                outcome = {'win': 1., 'draw': 0., 'loss': -1.}[view.result()]
                latest['reward'] = gamma ** (latest['duration_ticks'] - 1) * outcome
    if hold.active:
        hold.counts['released_match_end'] += 1
    if not records:
        raise RuntimeError('Match ended without a decision opportunity')
    assert records[-1]['done'] and sum(r['done'] for r in records) == 1
    assert all(r['duration_ticks'] >= 1 and sum(r['slot_mask']) > 1 for r in records)
    stats = dict(counts= dict(counts), holds=dict(hold.counts), cards_played=dict(cards),
                 deployment_statuses=dict(deployments), decision_elixir_sum=elixir_sum)
    return records, view.result(), stats


def aggregate_stats(stats):
    counts, holds, cards, deployments = Counter(), Counter(), Counter(), Counter()
    elixir = 0.
    for entry in stats:
        counts.update(entry['counts'])
        holds.update(entry['holds'])
        cards.update(entry['cards_played'])
        deployments.update(entry['deployment_statuses'])
        elixir += entry['decision_elixir_sum']
    decisions = counts['decisions']
    return dict(counts=dict(counts), holds=dict(holds), cards_played=dict(cards),
                deployment_statuses=dict(deployments), decision_elixir_sum=elixir,
                mean_decision_elixir=elixir / decisions if decisions else None,
                voluntary_wait_fraction=counts['voluntary_waits']/decisions if decisions else None,
                multi_card_fraction=counts['multi_card_decisions']/decisions if decisions else None)
