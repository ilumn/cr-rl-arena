import math

import cr_sim
import torch

from gandalf import (
    DECK, CARD_NAMES, NUM_TILES,
    encode_values, action_masks, decode_action,
)
from model_brains import ActorCritic, batch_inputs, select_action, evaluate_actions
from copy import deepcopy

battle = cr_sim.Battle()
battle.reset(seed=42, deck0=DECK, deck1=DECK)
obs = battle.view(0).observe()

encoded = encode_values(obs)
slot_mask, tile_masks = action_masks(obs)

model = ActorCritic(len(CARD_NAMES))

with torch.no_grad():
    slot_logits, tile_logits, values = model(
        batch_inputs([encoded])
    )

assert slot_logits.shape == (1, 5)
assert tile_logits.shape == (1, 4, NUM_TILES)
assert values.shape == (1,)
assert torch.isfinite(slot_logits).all()
assert torch.isfinite(tile_logits).all()
assert torch.isfinite(values).all()

for _ in range(100):
    slot, tile, log_probability, value = select_action(
        model, encoded, slot_mask, tile_masks
    )

    assert slot_mask[slot]
    assert math.isfinite(log_probability)
    assert math.isfinite(value)

    if slot == 0:
        assert tile is None
    else:
        assert tile_masks[slot - 1][tile]

    decode_action(slot, tile)

print("Gandalf network checks passed.")

for _ in range(20):
    slot, tile, old_log_prob, _ = select_action(
        model, encoded, slot_mask, tile_masks
    )

    with torch.no_grad():
        new_log_prob, entropy, _ = evaluate_actions(
            model,
            batch_inputs([encoded]),
            torch.tensor([slot]),
            torch.tensor([tile if tile is not None else 0]),
            torch.tensor([slot_mask], dtype=torch.bool),
            torch.tensor([tile_masks], dtype=torch.bool),
        )

    assert math.isclose(
        old_log_prob,
        new_log_prob.item(),
        abs_tol=1e-5,
    )
    assert torch.isfinite(entropy).all()

print("Joint action probabilities agree.")

forced_obs = deepcopy(obs)
forced_obs["elixir"] = 0.0
forced_obs["legal"] = [False] * 4

forced_input = encode_values(forced_obs)
forced_slots, forced_tiles = action_masks(forced_obs)

slot, tile, old_log_prob, _ = select_action(
    model, forced_input, forced_slots, forced_tiles
)

assert slot == 0
assert tile is None
assert abs(old_log_prob) < 1e-6

model.zero_grad(set_to_none=True)

log_probs, entropies, values = evaluate_actions(
    model,
    batch_inputs([forced_input]),
    torch.tensor([0]),
    torch.tensor([0]),  # Ignored placeholder for waiting.
    torch.tensor([forced_slots], dtype=torch.bool),
    torch.tensor([forced_tiles], dtype=torch.bool),
)

assert torch.isfinite(log_probs).all()
assert torch.isfinite(entropies).all()
assert torch.allclose(log_probs, torch.zeros_like(log_probs))
assert torch.allclose(entropies, torch.zeros_like(entropies))

# Confirm backpropagation through this case stays finite.
loss = values.square().mean() - log_probs.mean() - entropies.mean()
loss.backward()

for parameter in model.parameters():
    if parameter.grad is not None:
        assert torch.isfinite(parameter.grad).all()

model.zero_grad(set_to_none=True)
print("Forced-wait checks passed.")