"""Fixed-case screening evaluation; defaults to 20 openings in both seats."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from gandalf import CARD_NAMES
from model_brains import ActorCritic
from evaluate_gandalf import play_match
from train_gandalf_rollout import current_config, file_hash, code_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"] != current_config():
        raise ValueError("Checkpoint configuration differs from current code.")
    experiment = checkpoint.get("rollout_experiment")
    if experiment is not None and experiment["code_hashes"] != code_hashes():
        raise ValueError("Experiment code changed; evaluate with matching code.")

    model = ActorCritic(len(CARD_NAMES))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results, deployments = Counter(), Counter()
    seats = [Counter(), Counter()]
    cases = []
    started = time.perf_counter()

    for seed in range(args.seed_start, args.seed_start + args.games // 2):
        for seat in (0, 1):
            result, statuses = play_match(model, seed, seat)
            results[result] += 1
            seats[seat][result] += 1
            deployments.update(statuses)
            cases.append({"seed": seed, "seat": seat, "result": result})
            print(f"Game {len(cases)}/{args.games}: seed={seed}, seat={seat}, {result}", flush=True)

    report = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
        "seed_start": args.seed_start, "games": args.games,
        "results": dict(results), "win_rate": results["win"] / args.games,
        "seat_results": [dict(s) for s in seats],
        "deployment_statuses": dict(deployments), "cases": cases,
        "evaluation_seconds": time.perf_counter() - started,
        "branch_progress": checkpoint.get("branch_progress"),
        "code_hashes": code_hashes(),
    }
    print(f"\n{args.checkpoint}: {dict(results)}")
    print(f"Win rate: {100 * report['win_rate']:.1f}%")
    print(f"Seat results: {report['seat_results']}")
    print(f"Deployment statuses: {dict(deployments)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
