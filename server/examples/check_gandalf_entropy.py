"""Fast gradient/mask/PPO checks; no full matches or checkpoint writes."""

import contextlib
import copy
import io
from unittest.mock import patch

import cr_sim
import torch

import train_gandalf as core
from gandalf import CARD_NAMES, DECK, NUM_TILES, action_masks, encode_values
from gandalf_coordinate import CoordinateActorCritic
from gandalf_entropy import entropy_terms, separated_evaluator
from model_brains import batch_inputs, evaluate_actions, select_action


def main():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    slot_logits = torch.zeros(1, 5, dtype=torch.float64, requires_grad=True)
    tile_logits = (torch.randn(1, 4, NUM_TILES, dtype=torch.float64) * .5).requires_grad_()
    slots = torch.tensor([[True, True, True, False, False]])
    tiles = torch.zeros(1, 4, NUM_TILES, dtype=torch.bool)
    tiles[0, 0, :16] = True
    tiles[0, 1, :64] = True
    parts = entropy_terms(slot_logits, tile_logits, slots, tiles)
    joint_slot, joint_tile = torch.autograd.grad(
        parts['joint_tile'].sum(), (slot_logits, tile_logits), retain_graph=True)
    separated_slot, separated_tile = torch.autograd.grad(
        parts['separated_tile'].sum(), (slot_logits, tile_logits),
        allow_unused=True, retain_graph=True)
    assert joint_slot[0, 0] < -1e-6
    assert separated_slot is None or torch.count_nonzero(separated_slot) == 0
    assert torch.equal(parts['joint_tile'], parts['separated_tile'])
    assert torch.allclose(joint_tile, separated_tile, atol=1e-12)
    assert joint_tile.norm() > 0
    print(f'PASS direct gradient: joint tile-entropy gradient on wait logit='
          f'{joint_slot[0, 0].item():.6f}; separated=0')
    print('PASS placement exploration: same bonus value and same nonzero tile gradient')

    # No affordable cards: zero entropy and finite gradients, including dummy branches.
    wait_slots = torch.tensor([[True, False, False, False, False]])
    wait_tiles = torch.zeros_like(tiles)
    wait = entropy_terms(slot_logits, tile_logits, wait_slots, wait_tiles)
    assert all(torch.count_nonzero(v) == 0 for v in wait.values())
    grads = torch.autograd.grad(wait['slot'].sum() + wait['separated_tile'].sum(),
                                (slot_logits, tile_logits), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    print('PASS forced wait: zero bonus; finite gradients; empty tile branches safe')

    battle = cr_sim.Battle()
    battle.reset(seed=42, deck0=DECK, deck1=DECK)
    obs = battle.view(0).observe()
    encoded = encode_values(obs)
    slot_mask, tile_masks = action_masks(obs)
    model = CoordinateActorCritic(len(CARD_NAMES))
    inputs = batch_inputs([encoded])
    adapter = separated_evaluator(.01, .01, .01)
    records = []
    for i in range(8):
        slot, tile, log, value = select_action(model, encoded, slot_mask, tile_masks)
        args = (model, inputs, torch.tensor([slot]), torch.tensor([tile or 0]),
                torch.tensor([slot_mask]), torch.tensor([tile_masks]))
        original = evaluate_actions(*args)
        separated = adapter(*args)
        assert all(torch.equal(a, b) for a, b in zip(original, separated))
        assert abs(separated[0].item() - log) < 1e-5
        records.append(dict(observations=encoded, slot_action=slot, tile_id=tile,
                            log_probability=log, value_estimate=value,
                            slot_mask=slot_mask, tile_masks=tile_masks,
                            reward=1. if i % 2 else -1., done=True))
    # With the real model, shared features still receive placement gradients;
    # the direct tile-entropy gradient into slot_head must disappear.
    output = model(inputs)
    real = entropy_terms(output[0], output[1], torch.tensor([slot_mask]),
                         torch.tensor([tile_masks]))
    joint_grad = torch.autograd.grad(real['joint_tile'].sum(),
                                    model.slot_head.weight, retain_graph=True)[0]
    sep_grad = torch.autograd.grad(real['separated_tile'].sum(),
                                  model.slot_head.weight, allow_unused=True)[0]
    assert joint_grad.norm() > 0
    assert sep_grad is None or torch.count_nonzero(sep_grad) == 0
    print('PASS network: direct slot-head tile-entropy gradient removed; '
          'sampler log probabilities, values and entropy values unchanged')

    settings_before = dict(core.SETTINGS)
    returns, advantages = core.calculate_returns_and_advantages(records, gae_lambda=1.)
    for evaluator in [evaluate_actions, adapter]:
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=core.SETTINGS['learning_rate'])
        torch.manual_seed(456)
        with patch.dict(core.SETTINGS, {'ppo_epochs': 1, 'actor_batch_size': 8,
                                        'critic_batch_size': 8}), \
                patch.object(core, 'evaluate_actions', evaluator), \
                contextlib.redirect_stdout(io.StringIO()):
            core.ppo_update(candidate, optimizer, records, returns, advantages)
        assert all(torch.isfinite(p).all() for p in candidate.parameters())
        assert any(not torch.equal(p, dict(model.named_parameters())[name])
                   for name, p in candidate.named_parameters() if name.startswith('tile_head.'))
    assert dict(core.SETTINGS) == settings_before
    assert core.evaluate_actions is evaluate_actions
    print('PASS PPO: both modes update finite weights; settings/evaluator restored')
    print('All entropy checks passed. No agents, simulator or checkpoint files changed.')


if __name__ == '__main__':
    main()
