"""Fork a saved MC model for a fixed-experience rollout-size experiment."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from gandalf import (
    BOARD_HEIGHT, BOARD_WIDTH, CARD_NAMES, DECK, MAX_ENTITIES,
)
from model_brains import ActorCritic
from train_gandalf import (
    SETTINGS, calculate_returns_and_advantages, collect_match, ppo_update,
    report_exploration, report_rollout_quality,
)


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_config():
    return {
        "encoding_version": 2,
        "action_version": 2,
        "architecture": "entity_pool_slot_tile_v1",
        "card_names": CARD_NAMES,
        "deck": DECK,
        "max_entities": MAX_ENTITIES,
        "board_width": BOARD_WIDTH,
        "board_height": BOARD_HEIGHT,
        "gamma": SETTINGS["gamma"],
    }


def code_hashes():
    return {
        name: file_hash(Path(__file__).parent / name)
        for name in (
            "gandalf.py", "model_brains.py", "train_gandalf.py", "random_agent.py",
            "train_gandalf_rollout.py",
        )
    }


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target-choices", required=True, type=int)
    parser.add_argument("--decision-budget", type=int, default=4096)
    parser.add_argument("--run-seed", type=int, default=12345)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.target_choices <= 0:
        parser.error("--target-choices must be positive")
    if args.decision_budget <= 0:
        parser.error("--decision-budget must be positive")

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    starting_batch = source["completed_batches"]
    if source["config"] != current_config():
        raise ValueError("Source model/observation/action configuration differs.")
    setup = source.get("training_setup", {})
    if setup.get("estimator") != "mc" or setup.get("gae_lambda") != 1.0:
        raise ValueError("Expected the MC source checkpoint.")

    settings = dict(SETTINGS, target_choices=args.target_choices)
    saved_settings = source.get("experiment_metadata", {}).get("settings")
    if saved_settings is None:
        raise ValueError("Source checkpoint lacks detailed training settings.")

    experiment = {
        "version": 1,
        "source_sha256": file_hash(args.source),
        "source_batch": starting_batch,
        "estimator": "mc",
        "gae_lambda": 1.0,
        "settings": settings,
        "run_seed": args.run_seed,
        "decision_budget": args.decision_budget,
        "code_hashes": code_hashes(),
    }
    run_dir = args.run_dir or (
        ROOT / "checkpoints" / "gandalf" / "rollout_sweep_v1"
        / f"choices{args.target_choices}"
    )
    latest = run_dir / "latest.pt"

    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state.get("rollout_experiment") != experiment:
            raise ValueError("Existing branch has different settings/source/code; use a new directory.")
        progress = dict(state["branch_progress"])
        print(f"Resuming branch: {progress}", flush=True)
    else:
        state = source
        progress = {
            "meaningful_choices": 0, "simulator_ticks": 0,
            "completed_matches": 0, "updates": 0, "training_seconds": 0.0,
        }
        print(f"Forking starting batch into {run_dir}", flush=True)

    torch.manual_seed(args.run_seed)
    model = ActorCritic(len(CARD_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = settings["learning_rate"]
    next_seed = state["next_seed"]

    print(json.dumps(experiment, indent=2), flush=True)
    if args.dry_run:
        print("Dry run passed. No training or files written.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.json").write_text(json.dumps(experiment, indent=2) + "\n")

    with patch.dict(SETTINGS, settings):
        while progress["meaningful_choices"] < args.decision_budget:
            started = time.perf_counter()
            completed_batches = starting_batch + progress["updates"]
            collection_seed = args.run_seed + 10_000 * completed_batches
            random.seed(collection_seed)
            torch.manual_seed(collection_seed)
            batch, choices, results = [], 0, Counter()

            while choices < args.target_choices or sum(results.values()) < settings["minimum_matches"]:
                records, result = collect_match(
                    model, next_seed, seat=(next_seed - 1) % 2
                )
                next_seed += 1
                batch.extend(records)
                choices += sum(sum(t["slot_mask"]) > 1 for t in records)
                results[result] += 1

            print(
                f"Rollout target {args.target_choices}, update {progress['updates'] + 1}: "
                f"{len(batch)} ticks, {choices} choices, results={dict(results)}",
                flush=True,
            )
            returns, advantages = calculate_returns_and_advantages(
                batch, gamma=settings["gamma"], gae_lambda=1.0
            )
            report_rollout_quality(batch, returns, advantages)
            report_exploration(model, batch)
            ppo_update(model, optimizer, batch, returns, advantages)

            progress["meaningful_choices"] += choices
            progress["simulator_ticks"] += len(batch)
            progress["completed_matches"] += sum(results.values())
            progress["updates"] += 1
            progress["training_seconds"] += time.perf_counter() - started

            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": current_config(), "next_seed": next_seed,
                "completed_batches": starting_batch + progress["updates"],
                "training_setup": {
                    **setup, "target_choices": args.target_choices, "run_seed": args.run_seed,
                },
                "rollout_experiment": experiment, "branch_progress": dict(progress),
            }
            atomic_save(checkpoint, latest)
            print(f"Saved branch progress: {progress}", flush=True)

    # Full matches and rollouts can overshoot the decision budget; report reality.
    (run_dir / "summary.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(f"Finished: {latest}\nActual experience: {progress}", flush=True)


if __name__ == "__main__":
    main()
