import random
import sys
from collections import Counter
from pathlib import Path

import cr_sim
import torch

from gandalf import (
    DECK, CARD_NAMES, BOARD_WIDTH, BOARD_HEIGHT, MAX_ENTITIES,
    encode_values, action_masks, decode_action,
)
from model_brains import ActorCritic, select_action
from random_agent import policy as opponent_policy


@torch.no_grad()
def play_match(
    model, seed, seat, sampling_seed=None, metrics=None,
    opponent_fn=opponent_policy,
):
    if sampling_seed is None:
        sampling_seed = seed * 2 + seat
    # Reset randomness for repeatable comparisons.
    random.seed(sampling_seed)
    torch.manual_seed(sampling_seed)

    battle = cr_sim.Battle()
    battle.reset(seed=seed, deck0=DECK, deck1=DECK)

    gandalf_view = battle.view(seat)
    opponent_view = battle.view(1 - seat)
    deployments = Counter()

    while not battle.done():
        obs = gandalf_view.observe()
        encoded = encode_values(obs)
        slot_mask, tile_masks = action_masks(obs)

        slot_action, tile_id, _, _ = select_action(
            model, encoded, slot_mask, tile_masks
        )
        gandalf_action = decode_action(slot_action, tile_id)

        opponent_action = opponent_fn(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)

        # Battle.step expects actions in actual seat order.
        actions = [None, None]
        actions[seat] = gandalf_action
        actions[1 - seat] = opponent_action
        battle.step(*actions)

        if slot_action != 0:
            deployments[gandalf_view.observe()["last_action"]] += 1

    return gandalf_view.result(), deployments


def evaluate(model, label):
    model.eval()
    results = Counter()
    deployments = Counter()
    seat_results = [Counter(), Counter()]

    # Reserve these seeds for evaluation; don't train on them.
    for seed in range(200_000, 200_050):
        for seat in (0, 1):
            result, statuses = play_match(model, seed, seat)
            results[result] += 1
            seat_results[seat][result] += 1
            deployments.update(statuses)

    games = sum(results.values())
    rejected = sum(
        count for status, count in deployments.items()
        if status != "ok"
    )

    print(f"\n{label}: {games} matches")
    print("Results:", dict(results))
    print(f"Win rate: {100 * results['win'] / games:.1f}%")
    print("Seat 0:", dict(seat_results[0]))
    print("Seat 1:", dict(seat_results[1]))
    print("Deployment statuses:", dict(deployments))
    print("Rejected deployments:", rejected)


def main():
    root = Path(__file__).resolve().parents[2]
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else root / "checkpoints" / "gandalf" / "baseline_v2" / "latest.pt"
    )

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    expected_config = {
        "encoding_version": 2,
        "action_version": 2,
        "architecture": "entity_pool_slot_tile_v1",
        "card_names": CARD_NAMES,
        "deck": DECK,
        "max_entities": MAX_ENTITIES,
        "board_width": BOARD_WIDTH,
        "board_height": BOARD_HEIGHT,
        "gamma": 0.9999,
    }
    if checkpoint["config"] != expected_config:
        raise ValueError("Checkpoint and current encoding/config differ.")

    trained = ActorCritic(len(CARD_NAMES))
    trained.load_state_dict(checkpoint["model"])

    evaluate(
        trained,
        f"Trained: {checkpoint['completed_batches']} batches",
    )


if __name__ == "__main__":
    main()