import sys
from pathlib import Path

from train_gandalf import SETTINGS
from train_gandalf_rollout import main as run_experiment


ROOT = Path(__file__).resolve().parents[2]

# Accepted settings. Unlisted settings retain their shared defaults.
SETTINGS.update({
    "target_choices": 200,
    "learning_rate": .0006,
    "actor_batch_size": 64,
    "critic_batch_size": 128,
    "entropy_coefficient": 0.03,
})


if __name__ == "__main__":
    run_experiment([
        "--source", str(ROOT / "checkpoints/gandalf/latest.pt"),
        "--target-choices", str(SETTINGS["target_choices"]),
        "--run-dir", str(ROOT / "checkpoints/gandalf/candidate_B"),
        *sys.argv[1:],
    ])