import random
import torch

from gandalf import CARD_NAMES
from model_brains import ActorCritic
from train_gandalf import (
    collect_match,
    calculate_returns_and_advantages,
    ppo_update,
)


def main():
    random.seed(42)
    torch.manual_seed(42)

    model = ActorCritic(len(CARD_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    transitions, result = collect_match(model, seed=42)

    assert transitions
    assert transitions[-1]["done"]
    assert sum(t["done"] for t in transitions) == 1

    choices = sum(
        sum(t["slot_mask"]) > 1 for t in transitions
    )
    assert choices > 0

    for transition in transitions:
        slot = transition["slot_action"]
        tile = transition["tile_id"]

        assert transition["slot_mask"][slot]
        if slot != 0:
            assert tile is not None
            assert transition["tile_masks"][slot - 1][tile]

    returns, advantages = calculate_returns_and_advantages(transitions)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    ppo_update(model, optimizer, transitions, returns, advantages)

    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()

    for head in ("slot_head", "tile_head"):
        changed = any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if name.startswith(head + ".")
        )
        print(f"{head} changed: {changed}")

    print("Result:", result)
    print("Meaningful choices:", choices)
    print("Training integration check completed.")


if __name__ == "__main__":
    main()