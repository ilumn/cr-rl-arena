import torch
from torch import nn
from torch.distributions import Categorical

from gandalf import NUM_TILES

def evaluate_actions(
    model, inputs, slots, tiles, slot_masks, tile_masks,
    return_details=False,
):
    slot_logits, tile_logits, values = model(inputs)

    slot_distribution = Categorical(
        logits=slot_logits.masked_fill(
            ~slot_masks, float("-inf")
        )
    )

    # Unavailable cards have no legal tiles. Give those unused branches
    # one dummy tile so their distributions remain numerically valid.
    safe_tile_masks = tile_masks.clone()
    empty_branches = ~safe_tile_masks.any(dim=-1)
    safe_tile_masks[:, :, 0] |= empty_branches

    tile_distribution = Categorical(
        logits=tile_logits.masked_fill(
            ~safe_tile_masks, float("-inf")
        )
    )

    rows = torch.arange(slots.shape[0], device=slots.device)
    hand_slots = (slots - 1).clamp_min(0)
    playing = slots != 0

    selected_distribution = Categorical(
        logits=tile_distribution.logits[rows, hand_slots]
    )

    # For waiting, evaluate a harmless valid tile and discard its result.
    fallback_tiles = (
        safe_tile_masks[rows, hand_slots].long().argmax(dim=-1)
    )
    evaluated_tiles = torch.where(playing, tiles, fallback_tiles)

    tile_log_probs = selected_distribution.log_prob(evaluated_tiles)
    joint_log_probs = slot_distribution.log_prob(slots) + torch.where(
        playing,
        tile_log_probs,
        torch.zeros_like(tile_log_probs),
    )

    slot_entropy = slot_distribution.entropy()
    tile_entropy_per_card = tile_distribution.entropy()

    tile_entropy_contribution = (
        slot_distribution.probs[:, 1:]
        * tile_entropy_per_card
    ).sum(dim=-1)

    entropy = slot_entropy + tile_entropy_contribution

    if return_details:
        details = {
            "slot_entropy": slot_entropy,
            "tile_entropy_contribution": tile_entropy_contribution,
            "play_probability": slot_distribution.probs[:, 1:].sum(dim=-1),
        }
        return joint_log_probs, entropy, values, details

    return joint_log_probs, entropy, values

def batch_inputs(encoded_observations):
    """Convert a list of encoder dictionaries into batched tensors."""
    id_keys = {"entity_ids", "hand_ids", "next_card_id"}

    return {
        key: torch.tensor(
            [observation[key] for observation in encoded_observations],
            dtype=(
                torch.long if key in id_keys
                else torch.bool if key == "entity_mask"
                else torch.float32
            ),
        )
        for key in encoded_observations[0]
    }


class ActorCritic(nn.Module):
    def __init__(self, num_card_types):
        super().__init__()

        self.card_embedding = nn.Embedding(num_card_types, 16)

        # Shared across all entity slots.
        self.entity_encoder = nn.Sequential(
            nn.Linear(16 + 9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Entity summary + regions + hand + next card + globals.
        input_size = 64 + 64 + 4 * 18 + 17 + 15

        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.slot_head = nn.Linear(128, 5)
        self.value_head = nn.Linear(128, 1)

        # Tile scores depend on the board AND the specific hand card.
        self.tile_head = nn.Sequential(
            nn.Linear(128 + 18, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_TILES),
        )

    def forward(self, inputs):
        entity_embeddings = self.card_embedding(inputs["entity_ids"])
        entity_inputs = torch.cat(
            [entity_embeddings, inputs["entity_features"]], dim=-1
        )
        entity_hidden = self.entity_encoder(entity_inputs)

        mask = inputs["entity_mask"].unsqueeze(-1)
        entity_summary = (
            (entity_hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1)
        )

        hand = torch.cat(
            [
                self.card_embedding(inputs["hand_ids"]),
                inputs["hand_features"],
            ],
            dim=-1,
        )

        next_card = torch.cat(
            [
                self.card_embedding(inputs["next_card_id"]),
                inputs["next_card_cost"].unsqueeze(-1),
            ],
            dim=-1,
        )

        combined = torch.cat(
            [
                entity_summary,
                inputs["regions"],
                hand.flatten(start_dim=1),
                next_card,
                inputs["globals"],
            ],
            dim=-1,
        )
        hidden = self.shared(combined)

        slot_logits = self.slot_head(hidden)
        values = self.value_head(hidden).squeeze(-1)

        # Repeat board context for each of the four hand slots.
        board_context = hidden.unsqueeze(1).expand(-1, 4, -1)
        tile_inputs = torch.cat([board_context, hand], dim=-1)
        tile_logits = self.tile_head(tile_inputs)

        return slot_logits, tile_logits, values


@torch.no_grad()
def select_action(model, encoded, slot_mask, tile_masks):
    inputs = batch_inputs([encoded])
    slot_logits, tile_logits, values = model(inputs)

    slot_mask_tensor = torch.tensor(slot_mask, dtype=torch.bool)
    slot_distribution = Categorical(
        logits=slot_logits[0].masked_fill(
            ~slot_mask_tensor, float("-inf")
        )
    )

    slot = slot_distribution.sample()
    slot_action = slot.item()
    log_probability = slot_distribution.log_prob(slot)

    tile_id = None

    if slot_action != 0:
        hand_slot = slot_action - 1
        tile_mask = torch.tensor(
            tile_masks[hand_slot], dtype=torch.bool
        )

        tile_distribution = Categorical(
            logits=tile_logits[0, hand_slot].masked_fill(
                ~tile_mask, float("-inf")
            )
        )

        tile = tile_distribution.sample()
        tile_id = tile.item()
        log_probability += tile_distribution.log_prob(tile)

    return (
        slot_action,
        tile_id,
        log_probability.item(),
        values[0].item(),
    )