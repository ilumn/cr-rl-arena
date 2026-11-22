import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import cr_sim
import torch

from gandalf import (
    CARD_NAMES, DECK, encode_values, action_masks, decode_action,
)
from model_brains import ActorCritic, select_action
from random_agent import policy as random_policy
from train_gandalf_rollout import current_config, file_hash


def load_turk():
    path = Path(__file__).parent / "theMechanicalTurk_1.0.py"
    spec = importlib.util.spec_from_file_location(
        "evaluation_mechanical_turk", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MechanicalTurk from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def play_match(model, seed, seat, opponent_policy, opponent_deck):
    random.seed(seed * 2 + seat)
    torch.manual_seed(seed * 2 + seat)

    battle = cr_sim.Battle()
    decks = [None, None]
    decks[seat] = DECK
    decks[1 - seat] = opponent_deck
    battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])

    view = battle.view(seat)
    opponent_view = battle.view(1 - seat)
    statuses = Counter()

    while not battle.done():
        obs = view.observe()
        slots, tiles = action_masks(obs)
        slot, tile, _, _ = select_action(
            model, encode_values(obs), slots, tiles
        )
        action = decode_action(slot, tile)

        opponent_action = opponent_policy(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)

        actions = [None, None]
        actions[seat] = action
        actions[1 - seat] = opponent_action
        battle.step(*actions)

        if slot != 0:
            statuses[view.observe()["last_action"]] += 1

    return view.result(), statuses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-start", type=int, default=300_000)
    parser.add_argument(
        "--opponent", choices=("random", "turk"), default="random"
    )
    parser.add_argument("--games", type=int, default=40)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number (balanced seats)")
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint["config"] != current_config():
        raise ValueError("Checkpoint configuration differs from current code.")

    model = ActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.opponent == "turk":
        turk = load_turk()
        opponents = [("MechanicalTurk", turk.policy, turk.DECK)]
    else:
        opponents = [("Random", random_policy, DECK)]

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "seed_start": args.seed_start,
        "games": args.games,
        "opponent": args.opponent,
        "completed_batches": checkpoint.get("completed_batches"),
        "deck1_updates": checkpoint.get("deck1_updates"),
        "gandalf_deck": list(DECK),
        "opponents": {},
    }
    overall = Counter()

    for name, policy, deck in opponents:
        stage = (
            f"deck-one update {checkpoint['deck1_updates']}"
            if "deck1_updates" in checkpoint
            else f"batch {checkpoint.get('completed_batches', '?')}"
        )
        print(
            f"{args.checkpoint} ({stage})\n"
            f"Evaluating against {name}: {args.games} matches...",
            flush=True,
        )
        results = Counter()
        statuses = Counter()
        cases = []

        # Every opening is played in both seats.
        for seed in range(args.seed_start, args.seed_start + args.games // 2):
            for seat in (0, 1):
                result, feedback = play_match(
                    model, seed, seat, policy, deck
                )
                results[result] += 1
                statuses.update(feedback)
                cases.append({
                    "seed": seed, "seat": seat, "result": result,
                })
                if len(cases) % 10 == 0:
                    print(f"Completed {len(cases)}/{args.games}", flush=True)

        rejected = sum(
            count for status, count in statuses.items()
            if status != "ok"
        )
        games = sum(results.values())
        print(
            f"{name}: "
            f"{results['win']}W/{results['loss']}L/{results['draw']}D, "
            f"win rate {100 * results['win'] / games:.1f}%, "
            f"rejected deployments {rejected}",
            flush=True,
        )

        overall.update(results)
        report["opponents"][name] = {
            "deck": list(deck),
            "results": dict(results),
            "deployment_statuses": dict(statuses),
            "seat_results": [
                dict(Counter(c["result"] for c in cases if c["seat"] == seat))
                for seat in (0, 1)
            ],
            "cases": cases,
        }

    total_games = sum(overall.values())
    score = (
        overall["win"] + 0.5 * overall["draw"]
    ) / total_games
    print(
        f"Overall: {overall['win']}W/{overall['loss']}L/"
        f"{overall['draw']}D; score {100 * score:.1f}%"
    )
    report["overall"] = dict(overall)
    report["score"] = score

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(f"Saved results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
