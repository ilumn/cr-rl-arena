"""Small behavior audit, not a training job or a win-rate evaluation.

Replay selected known losses at the same sampling seeds. Records legal card
alternatives, deployment context, five-second troop movement, and Fireball casts.
Fireball damage uses a paired replay with the same cast redirected to a verified
empty tile: same cost/cycle/actions, identical board before impact. HP differences
at impact measure marginal effective damage, including same-tick overkill.
Low movement and target-capability flags are clues, not tactical correctness labels.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import cr_sim
import torch

from gandalf import (
    CARDS_PATH, DECK, MAX_ENTITIES, NUM_TILES, action_masks, card_definitions,
    decode_action, decode_tile, encode_values,
)
from gandalf_coordinate import PlacementTemperature
from gandalf_coordinate_battle import (
    atomic_save, exclusive_lock, file_hash, load_battle_model, load_turk, read_checkpoint,
)
from model_brains import batch_inputs, select_action


TILES = [decode_tile(i) for i in range(NUM_TILES)]
RADIUS = card_definitions['fireball']['radius']
FIVE_SECONDS = int(cr_sim.TICKS_PER_SECOND * 5)


def alive(entities, owner=None):
    return [e for e in entities if e[4] > 0 and (owner is None or e[0] == owner)]


def distance_to_entity(x, y, entity):
    # Engine splash rule: towers use box surfaces; ordinary troops use centers.
    dx = max(0., abs(x - entity[2]) - entity[10])
    dy = max(0., abs(y - entity[3]) - entity[11])
    return math.hypot(dx, dy)


def blast_targets(entities, x, y, padding=0.):
    return [e for e in alive(entities, 1)
            if distance_to_entity(x, y, e) <= RADIUS + padding]


def can_hit(card, enemy):
    target = card_definitions[card].get('targets', 'ground')
    if target == 'buildings':
        return bool(enemy[7] or card_definitions.get(enemy[1], {}).get('type') == 'building')
    return target == 'both' or (target == 'air' if enemy[6] else target == 'ground')


def behind_tower(entity, entities):
    if entity[6]:
        return False
    radius = card_definitions.get(entity[1], {}).get('collision_radius', .5)
    return any(entity[3] < tower[3] - tower[11]
               and abs(entity[2] - tower[2]) < tower[10] + radius + .25
               for tower in alive(entities, 0) if tower[7])


def calibrate_fireball():
    """Measure cast-to-impact timing in the installed binary, not a guessed fuse."""
    for seed in range(1, 65):
        battle = cr_sim.Battle()
        battle.reset(seed=seed, deck0=DECK, deck1=DECK)
        view = battle.view(0)
        obs = view.observe()
        if 'fireball' not in obs['hand']:
            continue
        tower = next(e for e in obs['entities'] if e[0] == 1 and e[7] and e[1] != 'king_tower')
        uid, hp = tower[9], tower[4]
        for step in range(1, 41):
            action = (obs['hand'].index('fireball'), tower[2], tower[3]) if step == 1 else None
            battle.step(action, None)
            current = {e[9]: e for e in view.observe()['entities']}
            if uid not in current or current[uid][4] < hp:
                return dict(steps_to_impact=step, isolated_tower_damage=hp - current.get(uid, [0]*5)[4])
    raise RuntimeError('Could not calibrate Fireball damage/timing with the installed simulator')


def decision_context(obs, slots, chosen, action, probabilities):
    entities = alive(obs['entities'])
    invaders = [e for e in entities if e[0] == 1 and not e[7] and e[3] < 15]
    towers = [e for e in entities if e[0] == 0 and e[7]]
    threat = min(invaders, key=lambda e: min(
        (distance_to_entity(e[2], e[3], t) for t in towers), default=e[3]), default=None)
    alternatives = [dict(slot=i, card=card, cost=obs['hand_cost'][i],
                         probability=probabilities[i+1],
                         can_hit_nearest_invader=can_hit(card, threat) if threat else None)
                    for i, card in enumerate(obs['hand']) if slots[i+1]]
    event = dict(tick=obs['tick'], elixir=obs['elixir'], hand=obs['hand'],
                 next_card=obs['next_card'], affordable_count=sum(obs['legal']),
                 alternatives=alternatives, wait_probability=probabilities[0],
                 chosen_card=obs['hand'][chosen-1] if chosen else 'wait',
                 action=action, invaders=invaders, nearest_invader=threat,
                 entities_at_decision=obs['entities'])
    if chosen:
        x, y = action[1:]
        event['distance_to_nearest_invader'] = min(
            (distance_to_entity(x, y, e) for e in invaders), default=None)
        event['same_lane_allies_ahead'] = [e for e in entities if e[0] == 0 and not e[7]
                                         and (e[2] < 9) == (x < 9) and e[3] > y + 1]
        event['same_lane_allies_across_river'] = [e for e in event['same_lane_allies_ahead'] if e[3] >= 17]
    return event


def finish_unit(track, current, tick, entities, match_ended=False):
    entry = dict(uid=track['uid'], card=track['card'], alive=current is not None,
                 observed_ticks=tick-track['tick'], path_length=track['path'],
                 initial_xy=track['start'], final_xy=list(current[2:4]) if current else None,
                 spawned_behind_tower=track['behind'], match_ended=match_ended)
    entry['low_movement_no_near_target'] = False
    if current is not None:
        distances = [distance_to_entity(current[2], current[3], e)
                     for e in alive(entities, 1) if can_hit(track['card'], e)]
        nearest = min(distances, default=None)
        entry['nearest_attackable_enemy_distance'] = nearest
        attack_range = card_definitions[track['card']].get('range', 0.)
        entry['low_movement_no_near_target'] = bool(
            entry['observed_ticks'] >= FIVE_SECONDS and not current[8]
            and track['path'] < .5 and (nearest is None or nearest > attack_range + 1.5))
    return entry


def entity_state(entities):
    return sorted((tuple(e) for e in entities), key=lambda e: e[9])


def fireball_damage(event, seed, seat, decks, actions):
    if 'impact_after' not in event:
        return dict(status='match_ended_before_impact')
    before, after = event['impact_before'], event['impact_after']
    # Avoid every nearby enemy and the opponent's potential same-tick spawn.
    avoid = list(before['entities']) + list(after['entities'])
    opponent_action = actions.get(before['tick'], [None, None])[1-seat]
    if opponent_action is not None:
        _, x, y = opponent_action
        x = 18 - (max(0, min(17, math.floor(x))) + .5)
        y = 32 - (max(0, min(31, math.floor(y))) + .5)
        avoid.append([1, 'knight', x, y, 1, 1, False, False, False, -1, 0., 0.])
    blank = next((p for p in TILES if not blast_targets(avoid, *p, padding=2.)), None)
    if blank is None:
        return dict(status='no_safe_empty_control_target')
    control = cr_sim.Battle()
    control.reset(seed=seed, deck0=decks[0], deck1=decks[1])
    view = control.view(seat)
    for tick in range(after['tick']):
        if tick == before['tick'] and entity_state(view.observe()['entities']) != entity_state(before['entities']):
            return dict(status='control_diverged_before_impact')
        pair = list(actions.get(tick, [None, None]))
        if tick == event['tick']:
            pair[seat] = (event['action'][0], *blank)
        control.step(*pair)
        if tick == event['tick'] and view.observe()['last_action'] != 'ok':
            return dict(status='control_cast_rejected')
    control_entities = view.observe()['entities']
    if blast_targets(control_entities, *blank, padding=1.):
        return dict(status='control_target_not_empty')
    original = {e[9]: e for e in alive(after['entities'], 1)}
    alternative = {e[9]: e for e in alive(control_entities, 1)}
    changes = []
    for uid in original.keys() | alternative.keys():
        a, b = original.get(uid), alternative.get(uid)
        difference = (b[4] if b else 0) - (a[4] if a else 0)
        if difference:
            entity = b if b else a
            changes.append(dict(uid=uid, card=entity[1], is_tower=entity[7], marginal_hp=difference))
    return dict(status='verified_paired_impact', control_target=blank, targets=changes,
                troop_hp=sum(c['marginal_hp'] for c in changes if not c['is_tower']),
                tower_hp=sum(c['marginal_hp'] for c in changes if c['is_tower']))


@torch.no_grad()
def audit_match(model, turk, seed, seat, timing):
    # Exactly the evaluation harness seeds/order; audit calculations draw no RNG.
    random.seed(seed*2+seat)
    torch.manual_seed(seed*2+seat)
    decks = [None, None]
    decks[seat], decks[1-seat] = DECK, turk.DECK
    battle = cr_sim.Battle()
    battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])
    view, other = battle.view(seat), battle.view(1-seat)
    decisions, movements, casts, monitors, pending, actions = [], [], [], [], [], {}
    counts, affordable, plays = Counter(), Counter(), Counter()
    while not battle.done():
        obs = view.observe()
        encoded = encode_values(obs)
        slots, tiles = action_masks(obs)
        chosen, tile, _, _ = select_action(model, encoded, slots, tiles)
        action = decode_action(chosen, tile)
        counts['ticks'] += 1
        counts['encoder_overflow_ticks'] += len(alive(obs['entities'])) > MAX_ENTITIES
        event = None
        if sum(slots) > 1:
            counts['meaningful_decision_ticks'] += 1
            affordable[sum(obs['legal'])] += 1
            counts['multi_card_decision_ticks'] += sum(obs['legal']) >= 2
            slot_logits, _, _ = model(batch_inputs([encoded]))
            probabilities = slot_logits[0].masked_fill(~torch.tensor(slots), -torch.inf).softmax(-1).tolist()
            event = decision_context(obs, slots, chosen, action, probabilities)
            decisions.append(event)
            counts['voluntary_waits'] += chosen == 0
        opponent_action = turk.policy(other.observe())
        pair = [None, None]
        pair[seat] = action
        pair[1-seat] = tuple(opponent_action) if opponent_action is not None else None
        if any(a is not None for a in pair):
            actions[obs['tick']] = pair
        battle.step(*pair)
        after = view.observe()
        if chosen:
            counts['deployments_selected'] += 1
            status = after['last_action']
            counts[f'deployment_{status}'] += 1
            event['status'] = status
            if status == 'ok':
                card = event['chosen_card']
                plays[card] += 1
                if card == 'fireball':
                    event['blast_at_cast'] = blast_targets(obs['entities'], *action[1:])
                    event['impact_tick'] = obs['tick'] + timing['steps_to_impact']
                    casts.append(event)
                    pending.append(event)
                elif card_definitions[card]['type'] == 'troop':
                    old_ids = {e[9] for e in obs['entities']}
                    for entity in alive(after['entities'], 0):
                        if entity[9] not in old_ids and entity[1] == card:
                            monitors.append(dict(uid=entity[9], card=card, tick=after['tick'],
                                                 start=list(entity[2:4]), last=list(entity[2:4]),
                                                 path=0., behind=behind_tower(entity, after['entities'])))
        survivors = []
        current = {e[9]: e for e in alive(after['entities'], 0)}
        for track in monitors:
            entity = current.get(track['uid'])
            if entity:
                track['path'] += math.dist(track['last'], entity[2:4])
                track['last'] = list(entity[2:4])
            if entity is None or after['tick'] - track['tick'] >= FIVE_SECONDS:
                movements.append(finish_unit(track, entity, after['tick'], after['entities']))
            else:
                survivors.append(track)
        monitors = survivors
        remaining = []
        for cast in pending:
            if after['tick'] == cast['impact_tick']:
                cast['impact_before'] = dict(tick=obs['tick'], entities=obs['entities'])
                cast['impact_after'] = dict(tick=after['tick'], entities=after['entities'])
                cast['blast_before_impact'] = blast_targets(obs['entities'], *cast['action'][1:])
            else:
                remaining.append(cast)
        pending = remaining
    final = view.observe()
    current = {e[9]: e for e in alive(final['entities'], 0)}
    movements.extend(finish_unit(t, current.get(t['uid']), final['tick'], final['entities'], True)
                     for t in monitors)
    for cast in casts:
        cast['damage_audit'] = fireball_damage(cast, seed, seat, decks, actions)
    return dict(seed=seed, seat=seat, result=view.result(), counters=dict(counts),
                affordable_histogram=dict(affordable), cards_played=dict(plays),
                decisions=decisions, troop_movement=movements,
                fireball_ticks=[c['tick'] for c in casts],
                replay=dict(seed=seed, deck0=decks[0], deck1=decks[1], frame='seat',
                            actions=[[tick, player, list(action)] for tick, pair in actions.items()
                                     for player, action in enumerate(pair) if action is not None]))


def summary(matches):
    counts, affordable, cards = Counter(), Counter(), Counter()
    hand_counts, opportunities, threat_flags, push_context = Counter(), Counter(), [], Counter()
    elixir = []
    movement, spells = [], []
    for match in matches:
        counts.update(match['counters'])
        affordable.update({int(k):v for k,v in match['affordable_histogram'].items()})
        cards.update(match['cards_played'])
        movement.extend(match['troop_movement'])
        spells.extend(d for d in match['decisions'] if d['chosen_card']=='fireball' and d.get('status')=='ok')
        for event in match['decisions']:
            elixir.append(event['elixir'])
            hand_counts.update(event['hand'])
            opportunities.update(a['card'] for a in event['alternatives'])
            selected = event['chosen_card']
            if selected == 'wait':
                continue
            if event['nearest_invader']:
                chosen = next(a for a in event['alternatives'] if a['card']==selected)
                if not chosen['can_hit_nearest_invader'] and any(
                        a['can_hit_nearest_invader'] for a in event['alternatives']):
                    threat_flags.append(dict(seed=match['seed'], seat=match['seat'],
                                             tick=event['tick'], chosen=selected,
                                             nearest_invader=event['nearest_invader'][1],
                                             alternatives=[a['card'] for a in event['alternatives']]))
            if card_definitions[selected]['type']=='troop':
                across = any(e[0]==0 and not e[7] and e[4]>0 and e[3]>=17
                             for e in event['entities_at_decision'])
                if across:
                    push_context['troop_plays_while_allies_across_river'] += 1
                    push_context['same_lane'] += bool(event['same_lane_allies_across_river'])
    completed = [m for m in movement if m['observed_ticks'] >= FIVE_SECONDS and m['alive']]
    verified = [s['damage_audit'] for s in spells if s['damage_audit']['status']=='verified_paired_impact']
    return dict(counters=dict(counts), affordable_histogram=dict(sorted(affordable.items())),
                cards_played=dict(cards), troop_units_observed=len(movement),
                card_opportunities={card:dict(in_hand_decision_ticks=hand_counts[card],
                                             deployable_decision_ticks=opportunities[card],
                                             times_played=cards[card])
                                    for card in DECK},
                elixir_at_decisions=dict(min=min(elixir), median=statistics.median(elixir),
                                        max=max(elixir)) if elixir else {},
                target_capability_review_flags=threat_flags, push_context=dict(push_context),
                troop_units_surviving_5s=len(completed),
                low_movement_no_near_target=sum(m['low_movement_no_near_target'] for m in completed),
                ground_units_spawned_behind_tower=sum(m['spawned_behind_tower'] for m in movement),
                fireballs=len(spells), verified_fireballs=len(verified),
                fireballs_zero_marginal_hp=sum(s['troop_hp']+s['tower_hp']==0 for s in verified),
                fireball_marginal_troop_hp=sum(s['troop_hp'] for s in verified),
                fireball_marginal_tower_hp=sum(s['tower_hp'] for s in verified))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--temperature', type=float, default=.25)
    parser.add_argument('--cases-from', type=Path, help='replay known losses from a matching evaluation report')
    parser.add_argument('--games', type=int, default=4, help='maximum matches to audit, not a win-rate sample')
    parser.add_argument('--seed-start', type=int, default=500_000)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.games <= 0 or not 0 < args.temperature < float('inf') or args.seed_start < 0:
        parser.error('games/temperature must be positive and finite; seed-start nonnegative')
    torch.set_num_threads(1)
    checkpoint, data = read_checkpoint(args.checkpoint)
    digest = hashlib.sha256(data).hexdigest()
    model = PlacementTemperature(load_battle_model(checkpoint), args.temperature).eval()
    turk = load_turk()
    opponent_hash = file_hash(Path(__file__).parent / 'theMechanicalTurk_1.0.py')
    simulator_hash = file_hash(cr_sim.__file__)
    if args.cases_from:
        evaluation = json.loads(args.cases_from.read_text())
        plan = evaluation['plan']
        if (plan['checkpoint_sha256'] != digest or plan['opponent_sha256'] != opponent_hash
                or plan['simulator_sha256'] != simulator_hash):
            raise ValueError('Evaluation report does not match this checkpoint/opponent/simulator')
        candidates = [c for c in evaluation['cases'] if c['policy']==f'T={args.temperature:g}' and c['result']=='loss']
        # Alternate seats where possible rather than picking only one perspective.
        queues = [[c for c in candidates if c['seat']==seat] for seat in (0,1)]
        cases=[]
        while len(cases)<args.games and any(queues):
            for queue in queues:
                if queue and len(cases)<args.games:
                    cases.append(queue.pop(0))
    else:
        cases=[dict(seed=args.seed_start+i//2, seat=i%2) for i in range(args.games)]
    if not cases:
        raise ValueError('No matching losses to audit')
    output = args.output_dir or args.checkpoint.parent / f'audit_{args.checkpoint.stem}_T{args.temperature:g}'
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(kind='gandalf_behavior_audit_v1', checkpoint_sha256=digest,
                       opponent_sha256=opponent_hash, simulator_sha256=simulator_hash,
                       code_sha256=file_hash(Path(__file__)), temperature=args.temperature,
                       policy_files={name:file_hash(Path(__file__).parent/name)
                                     for name in ['gandalf.py','model_brains.py','gandalf_coordinate.py']},
                       cards_sha256=file_hash(CARDS_PATH))
    with exclusive_lock(output/'.audit.lock'):
        timing=calibrate_fireball()
        print('Fireball timing calibration:',timing,flush=True)
        matches=[]
        for case in cases:
            path=output/f"seed_{case['seed']}_seat_{case['seat']}.json"
            if path.exists():
                match=json.loads(path.read_text())
                if match.get('audit_fingerprint')!=fingerprint:
                    raise ValueError('Cached audit has different settings/code; choose --output-dir')
            else:
                match=audit_match(model,turk,case['seed'],case['seat'],timing)
                if case.get('result') and match['result']!=case['result']:
                    raise RuntimeError('Audited result differs from the selected evaluation case')
                match['audit_fingerprint']=fingerprint
                atomic_save(match,path,json_format=True)
            matches.append(match)
            print(f"Audited seed={case['seed']} seat={case['seat']}: {match['result']}, "
                  f"{match['counters'].get('deployments_selected',0)} deployments",flush=True)
        report=dict(fingerprint=fingerprint,calibration=timing,
                    case_files=[f"seed_{c['seed']}_seat_{c['seat']}.json" for c in cases],
                    summary=summary(matches),
                    interpretation='Selected losses are a behavior sample, not a win-rate estimate. '
                    'Low movement can have legitimate causes; target capability is not counter quality.')
        atomic_save(report,output/'summary.json',json_format=True)
        print(json.dumps(report['summary'],indent=2),flush=True)
        print(f'Audit saved: {output}')


if __name__=='__main__':
    main()
