"""Small CPU checks for PPO direction and batch-boundary checkpoint resume.

Uses synthetic one-step tasks with real Gandalf observations, masks, sampler,
return calculation and PPO update. No full matches or user checkpoints are used.
"""

import io
import random
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cr_sim
import torch

from gandalf import CARD_NAMES, DECK, action_masks, encode_values
from model_brains import ActorCritic, batch_inputs, evaluate_actions, select_action
from train_gandalf import SETTINGS, calculate_returns_and_advantages, ppo_update


RUN_SEED = 31415


def make_model():
    model = ActorCritic(len(CARD_NAMES))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SETTINGS["learning_rate"]
    )
    return model, optimizer


def fixture(seed):
    battle = cr_sim.Battle()
    battle.reset(seed=seed, deck0=DECK, deck1=DECK)
    observation = battle.view(0).observe()
    slots, tiles = action_masks(observation)
    hand_slot = next(i for i in range(4) if slots[i + 1])
    available = [i for i, allowed in enumerate(tiles[hand_slot]) if allowed]
    assert len(available) >= 2

    # A deliberately tiny action space: wait, or this card at two legal tiles.
    slot_mask = [True] + [i == hand_slot for i in range(4)]
    tile_masks = [[False] * len(tiles[0]) for _ in range(4)]
    for tile in available[:2]:
        tile_masks[hand_slot][tile] = True
    return encode_values(observation), slot_mask, tile_masks, hand_slot + 1, available[:2]


@torch.no_grad()
def action_probability(model, encoded, slots, tiles, slot, tile):
    log_prob, _, _ = evaluate_actions(
        model,
        batch_inputs([encoded]),
        torch.tensor([slot]),
        torch.tensor([tile if tile is not None else 0]),
        torch.tensor([slots], dtype=torch.bool),
        torch.tensor([tiles], dtype=torch.bool),
    )
    return log_prob.exp().item()


@torch.no_grad()
def transition(model, encoded, slots, tiles, slot, tile, reward):
    log_prob, _, value = evaluate_actions(
        model,
        batch_inputs([encoded]),
        torch.tensor([slot]),
        torch.tensor([tile if tile is not None else 0]),
        torch.tensor([slots], dtype=torch.bool),
        torch.tensor([tiles], dtype=torch.bool),
    )
    return {
        "observations": encoded,
        "slot_action": slot,
        "tile_id": tile,
        "slot_mask": slots,
        "tile_masks": tiles,
        "log_probability": log_prob.item(),
        "value_estimate": value.item(),
        "reward": reward,
        "done": True,
    }


def update(model, optimizer, records):
    returns, advantages = calculate_returns_and_advantages(
        records, gamma=SETTINGS["gamma"], gae_lambda=1.0
    )
    with redirect_stdout(io.StringIO()):
        ppo_update(model, optimizer, records, returns, advantages)
    assert all(torch.isfinite(p).all() for p in model.parameters())


def check_reward_direction():
    random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    model, _ = make_model()
    encoded, slots, tiles, play_slot, (good_tile, bad_tile) = fixture(42)

    # Isolate the reward-based actor gradient from critic/entropy/shared effects.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("slot_head.") or name.startswith("tile_head.")
        )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=SETTINGS["learning_rate"],
    )
    records = [
        transition(model, encoded, slots, tiles, play_slot, good_tile, 1.0),
        transition(model, encoded, slots, tiles, play_slot, bad_tile, -1.0),
        transition(model, encoded, slots, tiles, 0, None, -1.0),
    ]
    before = action_probability(model, encoded, slots, tiles, play_slot, good_tile)
    with patch.dict(SETTINGS, {"entropy_coefficient": 0.0}):
        update(model, optimizer, records)
    after = action_probability(model, encoded, slots, tiles, play_slot, good_tile)
    assert after > before + 1e-7, (before, after)
    print(f"PASS reward direction: rewarded action {before:.6f} -> {after:.6f}")


def synthetic_batch(model, completed_batches, next_seed):
    # Match the trainer's batch-boundary reseeding convention.
    collection_seed = RUN_SEED + 10_000 * completed_batches
    random.seed(collection_seed)
    torch.manual_seed(collection_seed)
    records = []
    for seed in range(next_seed, next_seed + 3):
        encoded, slots, tiles, play_slot, targets = fixture(seed)
        rewarded_tile = random.choice(targets)
        for _ in range(4):
            slot, tile, log_prob, value = select_action(model, encoded, slots, tiles)
            reward = 1.0 if slot == play_slot and tile == rewarded_tile else -1.0
            records.append({
                "observations": encoded,
                "slot_action": slot,
                "tile_id": tile,
                "slot_mask": slots,
                "tile_masks": tiles,
                "log_probability": log_prob,
                "value_estimate": value,
                "reward": reward,
                "done": True,
            })
    return records, next_seed + 3


def assert_same(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def check_resume():
    torch.manual_seed(RUN_SEED)
    model, optimizer = make_model()
    records, next_seed = synthetic_batch(model, completed_batches=0, next_seed=1)
    update(model, optimizer, records)

    # Save the same state fields that the trainer restores. No RNG snapshot is
    # needed here because the real trainer reseeds at every batch boundary.
    with tempfile.TemporaryDirectory(prefix="gandalf-check-") as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_batches": 1,
            "next_seed": next_seed,
        }, path)

        uninterrupted, uninterrupted_seed = synthetic_batch(model, 1, next_seed)
        update(model, optimizer, uninterrupted)

        # Simulate a new process constructing a fresh model and optimizer.
        torch.manual_seed(999)
        random.seed(999)
        resumed_model, resumed_optimizer = make_model()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        resumed_model.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])

        resumed, resumed_seed = synthetic_batch(
            resumed_model, checkpoint["completed_batches"], checkpoint["next_seed"]
        )
        assert_same(uninterrupted, resumed, "rollout")
        update(resumed_model, resumed_optimizer, resumed)

        assert uninterrupted_seed == resumed_seed
        assert_same(model.state_dict(), resumed_model.state_dict(), "model")
        assert_same(optimizer.state_dict(), resumed_optimizer.state_dict(), "optimizer")

        # Negative control: prove restoring only weights is not sufficient.
        cold_model, cold_optimizer = make_model()
        cold_model.load_state_dict(checkpoint["model"])
        cold_records, _ = synthetic_batch(cold_model, 1, checkpoint["next_seed"])
        update(cold_model, cold_optimizer, cold_records)
        assert any(
            not torch.equal(parameter, cold_model.state_dict()[name])
            for name, parameter in model.state_dict().items()
        ), "Negative control failed: optimizer restoration was not exercised"

    print("PASS resume: identical next rollout, model and optimizer states")
    print("PASS negative control: omitting optimizer state changes the result")


def main():
    original_threads = torch.get_num_threads()
    try:
        # Small CPU fixtures are faster and easier to reproduce with one thread.
        torch.set_num_threads(1)
        check_reward_direction()
        check_resume()
    finally:
        torch.set_num_threads(original_threads)
    print("All Gandalf learning checks passed. No trained checkpoints modified.")


if __name__ == "__main__":
    main()
