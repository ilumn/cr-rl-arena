import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

import torch

# Import A's accepted settings without executing its training command.
from train_gandalf_A import SETTINGS
from train_gandalf import (
    collect_match,
    calculate_returns_and_advantages,
    ppo_update,
    report_rollout_quality,
    report_exploration,
)
from train_gandalf_rollout import current_config
from gandalf import CARD_NAMES, DECK
from model_brains import ActorCritic


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_turk():
    path = Path(__file__).parent / "theMechanicalTurk_1.0.py"
    spec = importlib.util.spec_from_file_location(
        "deck1_mechanical_turk", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "checkpoints/gandalf/deck1_turk_v1",
    )
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--run-seed", type=int, default=12345)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batches <= 0:
        parser.error("--batches must be positive")

    turk, turk_path = load_turk()
    source = torch.load(
        args.source, map_location="cpu", weights_only=True
    )

    if source["config"] != current_config():
        raise ValueError(
            "Starting checkpoint differs from the current model configuration."
        )

    # Record the implementation as well as the numerical settings.
    code_files = [
        Path(__file__),
        Path(__file__).parent / "gandalf.py",
        Path(__file__).parent / "model_brains.py",
        Path(__file__).parent / "train_gandalf.py",
        Path(__file__).parent / "train_gandalf_A.py",
        turk_path,
    ]

    experiment = {
        "name": "deck1_mechanical_turk",
        "source_sha256": file_hash(args.source),
        "settings": dict(SETTINGS),
        "estimator": "mc",
        "gae_lambda": 1.0,
        "gandalf_deck": list(DECK),
        "opponent_deck": list(turk.DECK),
        "run_seed": args.run_seed,
        "code_hashes": {
            path.name: file_hash(path) for path in code_files
        },
    }

    latest = args.run_dir / "latest.pt"

    if latest.exists():
        checkpoint = torch.load(
            latest, map_location="cpu", weights_only=True
        )
        if checkpoint.get("deck1_experiment") != experiment:
            raise ValueError(
                "This directory belongs to different settings or code. "
                "Use a new directory."
            )

        completed = checkpoint["deck1_updates"]
        print(f"Resuming deck-one training after {completed} updates.")
    else:
        checkpoint = source
        completed = 0
        print(
            "Starting deck-one continuation from "
            f"batch {source['completed_batches']}."
        )

    torch.manual_seed(args.run_seed)
    random.seed(args.run_seed)

    model = ActorCritic(len(CARD_NAMES))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=SETTINGS["learning_rate"]
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for group in optimizer.param_groups:
        group["lr"] = SETTINGS["learning_rate"]

    next_seed = checkpoint["next_seed"]

    print(json.dumps(experiment, indent=2))

    if args.dry_run:
        print("Validation passed. No training or files written.")
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the exact starting model for later comparison.
    initial_path = args.run_dir / "initial.pt"
    if not initial_path.exists():
        save_checkpoint(source, initial_path)

    while completed < args.batches:
        batch_seed = (
            args.run_seed
            + 10_000 * (source["completed_batches"] + completed)
        )
        random.seed(batch_seed)
        torch.manual_seed(batch_seed)

        batch = []
        choices = 0
        results = Counter()
        opponent_feedback = Counter()

        def opponent(obs):
            # Records feedback about the opponent's previous action.
            if obs["tick"] > 0:
                opponent_feedback[obs["last_action"]] += 1
            return turk.policy(obs)

        while (
            choices < SETTINGS["target_choices"]
            or sum(results.values()) < SETTINGS["minimum_matches"]
        ):
            records, result = collect_match(
                model,
                seed=next_seed,
                seat=(next_seed - 1) % 2,
                opponent_fn=opponent,
                opponent_deck=turk.DECK,
            )
            next_seed += 1
            batch.extend(records)
            results[result] += 1
            choices += sum(
                sum(t["slot_mask"]) > 1 for t in records
            )

        print(
            f"\nDeck-one update {completed + 1}/{args.batches}: "
            f"{len(batch)} ticks, {choices} choices, "
            f"results={dict(results)}",
            flush=True,
        )
        print(
            "MechanicalTurk observed action feedback:",
            dict(opponent_feedback),
        )

        returns, advantages = calculate_returns_and_advantages(
            batch,
            gamma=SETTINGS["gamma"],
            gae_lambda=1.0,
        )

        report_rollout_quality(batch, returns, advantages)
        report_exploration(model, batch)
        ppo_update(model, optimizer, batch, returns, advantages)

        completed += 1

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": current_config(),
            "next_seed": next_seed,
            "completed_batches": source["completed_batches"] + completed,
            "deck1_updates": completed,
            "deck1_experiment": experiment,
        }

        save_checkpoint(checkpoint, latest)

        if completed % 25 == 0 or completed == args.batches:
            snapshot = args.run_dir / f"update_{completed}.pt"
            save_checkpoint(checkpoint, snapshot)

        print(
            f"Saved deck-one update {completed}/{args.batches}.",
            flush=True,
        )


if __name__ == "__main__":
    main()