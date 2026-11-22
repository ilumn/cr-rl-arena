"""Opt-in entropy ablation. Sampling and joint PPO log probabilities stay intact.

Joint tile exploration: sum P(card) * H(tile | card).
Separated tile exploration: sum stop_gradient(P(card)) * H(tile | card).

Do not normalize by P(play): retaining the original weights preserves the tile
gradient and bonus value at a given policy. Only the direct gradient through
card probabilities is removed. Shared-layer updates can still affect both heads.
"""

import torch
from torch.distributions import Categorical

from model_brains import evaluate_actions as original_evaluate_actions


def entropy_terms(slot_logits, tile_logits, slot_masks, tile_masks):
    slot_distribution = Categorical(logits=slot_logits.masked_fill(~slot_masks, -torch.inf))
    safe_masks = tile_masks.clone()
    safe_masks[:, :, 0] |= ~safe_masks.any(dim=-1)
    tile_distribution = Categorical(logits=tile_logits.masked_fill(~safe_masks, -torch.inf))
    card_probabilities = slot_distribution.probs[:, 1:]
    per_card_entropy = tile_distribution.entropy()
    return {
        'slot': slot_distribution.entropy(),
        'joint_tile': (card_probabilities * per_card_entropy).sum(-1),
        'separated_tile': (card_probabilities.detach() * per_card_entropy).sum(-1),
    }


def separated_evaluator(slot_coefficient, tile_coefficient, loss_scale):
    """Adapter for existing PPO; no edits to the accepted trainer are needed."""
    if loss_scale <= 0:
        raise ValueError('Entropy loss scale must be positive')

    def evaluate(model, inputs, slots, tiles, slot_masks, tile_masks, return_details=False):
        outputs = model(inputs)
        # Reuse precisely the existing masked log-probability implementation,
        # including waiting and empty tile branches, without a second forward.
        log_probs, _, values, details = original_evaluate_actions(
            lambda _: outputs, inputs, slots, tiles, slot_masks, tile_masks,
            return_details=True)
        parts = entropy_terms(outputs[0], outputs[1], slot_masks, tile_masks)
        bonus = ((slot_coefficient / loss_scale) * parts['slot']
                 + (tile_coefficient / loss_scale) * parts['separated_tile'])
        if return_details:
            return log_probs, bonus, values, details
        return log_probs, bonus, values

    return evaluate
