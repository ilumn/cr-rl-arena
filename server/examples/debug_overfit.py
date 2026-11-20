import random
from collections import Counter

import cr_sim
import torch

from HAL9000 import DECK, NUM_ACTIONS, encode_observation
from hal_brains import ActorCritic
from train_hal import (
    collect_match,
    calculate_returns_and_advantages,
    ppo_update,
)
from evaluate_hal import play_match


TRAIN_SEEDS = [7000, 7001, 7002, 7003]


def check_fixed_seeds(model):
    was_training = model.training
    model.eval()

    results = Counter()
    deployments = Counter()
    metrics = Counter()

    for opening_index, seed in enumerate(TRAIN_SEEDS):
        opening_results = Counter()

        for repeat in range(10):
            sampling_seed = 20_000 + opening_index * 100 + repeat

            result, statuses = play_match(
                model,
                seed=seed,
                seat=0,
                sampling_seed=sampling_seed,
                metrics=metrics,
            )

            results[result] += 1
            opening_results[result] += 1
            deployments.update(statuses)

        print(f"Opening {seed}: {dict(opening_results)}")

    games = sum(results.values())
    rejected = sum(
        count
        for status, count in deployments.items()
        if status != "ok"
    )

    print("Overall:", dict(results))
    print(f"Win rate: {100 * results['win'] / games:.1f}%")
    print("Rejected deployments:", rejected)

    if metrics["choice_ticks"]:
        entropy = metrics["entropy_sum"] / metrics["choice_ticks"]
        print(f"Choice-only entropy: {entropy:.4f}")

    model.train(was_training)

def main():
    torch.manual_seed(12345)
    random.seed(12345)

    battle = cr_sim.Battle()
    battle.reset(seed=0, deck0=DECK, deck1=DECK)
    features = encode_observation(battle.view(0).observe())

    model = ActorCritic(len(features), NUM_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("Before training:")
    check_fixed_seeds(model)

    for update in range(30):
        batch = []

        # Fresh trajectories from the same small set of openings.
        # Vary action randomness on each collection pass.
        random.seed(10_000 + update)
        torch.manual_seed(10_000 + update)

        for seed in TRAIN_SEEDS:
            transitions, _ = collect_match(model, seed)
            batch.extend(transitions)

        returns, advantages = calculate_returns_and_advantages(batch)
        ppo_update(model, optimizer, batch, returns, advantages)

        if (update + 1) % 10 == 0:
            print(f"After {update + 1} updates:")
            check_fixed_seeds(model)


if __name__ == "__main__":
    main()