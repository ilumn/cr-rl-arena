import random
import sys
from collections import Counter
from pathlib import Path

import cr_sim
import torch

from HAL9000 import (
    DECK, ENTITY_TYPES, POSITIONS, NUM_ACTIONS,
    encode_observation, action_mask, decode_action,
)
from hal_brains import ActorCritic
from random_agent import policy as opponent_policy


@torch.no_grad()
def play_match(model, seed, seat, sampling_seed=None, metrics=None):
    if sampling_seed is None:
        sampling_seed = seed * 2 + seat
    # Reset randomness for repeatable comparisons.
    random.seed(sampling_seed)
    torch.manual_seed(sampling_seed)

    battle = cr_sim.Battle()
    battle.reset(seed=seed, deck0=DECK, deck1=DECK)

    hal_view = battle.view(seat)
    opponent_view = battle.view(1 - seat)
    deployments = Counter()

    while not battle.done():
        obs = hal_view.observe()
        features = torch.tensor(
            encode_observation(obs), dtype=torch.float32
        )
        mask = torch.tensor(action_mask(obs), dtype=torch.bool)

        logits, _ = model(features)
        logits = logits.masked_fill(~mask, float("-inf"))

        distribution = torch.distributions.Categorical(logits=logits)
        if metrics is not None and mask.sum().item() > 1:
            metrics["entropy_sum"] += distribution.entropy().item()
            metrics["choice_ticks"] += 1
        action_id = distribution.sample().item()
        hal_action = decode_action(action_id)

        opponent_action = opponent_policy(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)

        # Battle.step expects actions in actual seat order.
        actions = [None, None]
        actions[seat] = hal_action
        actions[1 - seat] = opponent_action
        battle.step(*actions)

        if action_id != 0:
            deployments[hal_view.observe()["last_action"]] += 1

    return hal_view.result(), deployments


def evaluate(model, label):
    model.eval()
    results = Counter()
    deployments = Counter()
    seat_results = [Counter(), Counter()]

    # Reserve these seeds for evaluation; don't train on them.
    for seed in range(100_000, 100_100):
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
        else root / "checkpoints" / "hal_batch9.pt"
    )

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    battle = cr_sim.Battle()
    battle.reset(seed=0, deck0=DECK, deck1=DECK)
    input_size = len(encode_observation(battle.view(0).observe()))

    expected_config = {
        "encoding_version": 1,
        "input_size": input_size,
        "num_actions": NUM_ACTIONS,
        "deck": DECK,
        "entity_types": ENTITY_TYPES,
        "positions": POSITIONS,
        "gamma": 0.9999,
    }
    if checkpoint["config"] != expected_config:
        raise ValueError("Checkpoint and current encoding/config differ.")

    trained = ActorCritic(input_size, NUM_ACTIONS)
    trained.load_state_dict(checkpoint["model"])

    # A fixed fresh network, not your original pre-training weights.
    torch.manual_seed(12345)
    untrained = ActorCritic(input_size, NUM_ACTIONS)

    evaluate(untrained, "Untrained baseline")
    evaluate(
        trained,
        f"Trained: {checkpoint['completed_batches']} batches",
    )


if __name__ == "__main__":
    main()