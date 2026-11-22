# Create and reset a battle
# imports - bring in tensor, neural network operations, ActorCritic, battle simulator, opponent_policy
import torch
import random
import argparse
from collections import Counter
from pathlib import Path
import cr_sim
from gandalf import (
    DECK, 
    CARD_NAMES, 
    BOARD_WIDTH, 
    BOARD_HEIGHT,
    MAX_ENTITIES, 
    encode_values, 
    action_masks, 
    decode_action,
    PLACEMENT_MARGIN,
)

from model_brains import ActorCritic, select_action, batch_inputs, evaluate_actions
from random_agent import policy as opponent_policy

# CHECKPOINT_PATH = (
#     Path(__file__).resolve().parents[2]
#     / "checkpoints"
#     / "gandalf"
#     / "larger_rollout_v1"
#     / "latest.pt"
# )

SETTINGS = {
    "learning_rate": 3e-4,
    "gamma": 0.9999,
    "target_choices": 128,
    "minimum_matches": 8,
    "actor_batch_size": 64,
    "critic_batch_size": 128,
    "ppo_epochs": 4,
    "clip_range": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "target_kl": 0.02,
    "max_grad_norm": 0.5,
    "placement_filter": "structure_and_rear_corridor_v1",
    "placement_margin": PLACEMENT_MARGIN,
}

@torch.no_grad()
def report_exploration(model, transitions):
    records = [
        t for t in transitions
        if sum(t["slot_mask"]) > 1
    ]

    if not records:
        print("No voluntary decisions to summarize.")
        return

    slot_entropy_sum = 0.0
    tile_entropy_sum = 0.0
    play_probability_sum = 0.0

    for start in range(0, len(records), 128):
        chunk = records[start:start + 128]

        _, _, _, details = evaluate_actions(
            model,
            batch_inputs([t["observations"] for t in chunk]),
            torch.tensor(
                [t["slot_action"] for t in chunk],
                dtype=torch.long,
            ),
            torch.tensor(
                [
                    t["tile_id"] if t["tile_id"] is not None else 0
                    for t in chunk
                ],
                dtype=torch.long,
            ),
            torch.tensor(
                [t["slot_mask"] for t in chunk],
                dtype=torch.bool,
            ),
            torch.tensor(
                [t["tile_masks"] for t in chunk],
                dtype=torch.bool,
            ),
            return_details=True,
        )

        slot_entropy_sum += details["slot_entropy"].sum().item()
        tile_entropy_sum += (
            details["tile_entropy_contribution"].sum().item()
        )
        play_probability_sum += (
            details["play_probability"].sum().item()
        )

    waits = sum(t["slot_action"] == 0 for t in records)
    deployments = len(records) - waits

    fireball_id = CARD_NAMES.index("fireball")
    fireball_opportunities = 0
    fireball_plays = 0

    for transition in records:
        hand = transition["observations"]["hand_ids"]
        slot_mask = transition["slot_mask"]

        if any(
            card == fireball_id and slot_mask[index + 1]
            for index, card in enumerate(hand)
        ):
            fireball_opportunities += 1

        slot = transition["slot_action"]
        if slot != 0 and hand[slot - 1] == fireball_id:
            fireball_plays += 1

    print(
        f"Slot entropy: {slot_entropy_sum / len(records):.4f}"
    )

    if play_probability_sum > 1e-8:
        print(
            "Tile entropy conditional on playing: "
            f"{tile_entropy_sum / play_probability_sum:.4f}"
        )

    print(
        f"Voluntary waits: {waits}/{len(records)} "
        f"({100 * waits / len(records):.1f}%)"
    )
    print(f"Deployments selected: {deployments}")
    print(
        f"Fireball selections: {fireball_plays}; "
        f"affordable Fireball decision ticks: "
        f"{fireball_opportunities}"
    )

def calculate_returns_and_advantages(
    transitions,
    gamma=SETTINGS["gamma"],
    gae_lambda=0.9995,
):
    if not transitions:
        raise ValueError("Cannot calculate GAE for an empty rollout.")

    # This implementation expects complete matches.
    assert transitions[-1]["done"]

    advantages = [0.0] * len(transitions)
    running_advantage = 0.0
    next_value = 0.0

    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        value = transition["value_estimate"]

        # A terminal transition must not bootstrap from the next match.
        continuation = 0.0 if transition["done"] else 1.0

        delta = (
            transition["reward"]
            + gamma * continuation * next_value
            - value
        )

        running_advantage = (
            delta
            + gamma * gae_lambda * continuation * running_advantage
        )

        advantages[index] = running_advantage
        next_value = value

    advantages = torch.tensor(advantages, dtype=torch.float32)
    values = torch.tensor(
        [t["value_estimate"] for t in transitions],
        dtype=torch.float32,
    )

    # Critic targets consistent with the GAE estimates.
    returns = advantages + values

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()

    return returns, advantages

@torch.no_grad()
def measure_policy_kl(model, transitions):
    records = [
        t for t in transitions
        if sum(t["slot_mask"]) > 1
    ]

    if not records:
        return 0.0

    total = 0.0

    for start in range(0, len(records), 128):
        chunk = records[start:start + 128]

        log_probs, _, _ = evaluate_actions(
            model,
            batch_inputs([t["observations"] for t in chunk]),
            torch.tensor(
                [t["slot_action"] for t in chunk], dtype=torch.long
            ),
            torch.tensor(
                [
                    t["tile_id"] if t["tile_id"] is not None else 0
                    for t in chunk
                ],
                dtype=torch.long,
            ),
            torch.tensor(
                [t["slot_mask"] for t in chunk], dtype=torch.bool
            ),
            torch.tensor(
                [t["tile_masks"] for t in chunk], dtype=torch.bool
            ),
        )

        old = torch.tensor(
            [t["log_probability"] for t in chunk],
            dtype=torch.float32,
        )
        log_ratio = log_probs - old
        estimate = log_ratio.exp() - 1 - log_ratio

        assert torch.isfinite(estimate).all()
        total += estimate.sum().item()

    return total / len(records)

def report_rollout_quality(transitions, returns, advantages):
    values = torch.tensor(
        [t["value_estimate"] for t in transitions],
        dtype=torch.float32,
    )
    choices = torch.tensor([
        sum(t["slot_mask"]) > 1
        for t in transitions
    ])

    target_variance = returns.var(unbiased=False)
    prediction_error = values - returns

    critic_mse = prediction_error.square().mean()
    critic_rmse = critic_mse.sqrt()

    print(
        f"Return variance: {target_variance.item():.6f}, "
        f"critic MSE: {critic_mse.item():.4f}, "
        f"critic RMSE: {critic_rmse.item():.4f}"
    )

    if target_variance > 1e-8:
        explained_variance = (
            1
            - (returns - values).var(unbiased=False)
            / target_variance
        )
        print(
            f"Critic explained variance: "
            f"{explained_variance.item():.4f}"
        )
    else:
        print("Critic explained variance: undefined (low target variance)")

    print(
        f"Critic prediction mean: {values.mean().item():.4f}, "
        f"return mean: {returns.mean().item():.4f}"
    )

    if choices.any():
        selected = advantages[choices]
        print(
            f"Raw choice advantages: "
            f"mean={selected.mean().item():.4f}, "
            f"std={selected.std(unbiased=False).item():.4f}, "
            f"positive={(selected > 0).float().mean().item():.1%}"
        )

def ppo_update(model, optimizer, transitions, returns, advantages):
    actor_indices = torch.tensor([
        index
        for index, transition in enumerate(transitions)
        if sum(transition["slot_mask"]) > 1
    ], dtype=torch.long)

    if len(actor_indices) == 0:
        print("No meaningful choices in this batch.")
        return

    returns = returns.detach()
    advantages = advantages.detach()

    # Normalize over all meaningful choices in this rollout.
    selected_advantages = advantages[actor_indices]
    normalized_advantages = (
        (selected_advantages - selected_advantages.mean())
        / (selected_advantages.std(unbiased=False) + 1e-8)
    )

    actor_batch_size = SETTINGS["actor_batch_size"]
    critic_batch_size = SETTINGS["critic_batch_size"]
    target_kl = SETTINGS["target_kl"]
    step_count = 0

    # Collection and recomputation should initially agree.
    initial_kl = measure_policy_kl(model, transitions)
    assert initial_kl < 1e-5, (
        f"Collection/update probability mismatch: KL={initial_kl}"
    )

    for epoch in range(SETTINGS["ppo_epochs"]):
        order = torch.randperm(len(actor_indices))

        for start in range(0, len(order), actor_batch_size):
            positions = order[start:start + actor_batch_size]
            indices = actor_indices[positions]
            records = [
                transitions[index] for index in indices.tolist()
            ]

            # ----- Actor: only meaningful decisions -----
            slot_actions = torch.tensor(
                [t["slot_action"] for t in records],
                dtype=torch.long,
            )
            tile_actions = torch.tensor(
                [
                    t["tile_id"] if t["tile_id"] is not None else 0
                    for t in records
                ],
                dtype=torch.long,
            )
            slot_masks = torch.tensor(
                [t["slot_mask"] for t in records],
                dtype=torch.bool,
            )
            tile_masks = torch.tensor(
                [t["tile_masks"] for t in records],
                dtype=torch.bool,
            )
            old_log_probs = torch.tensor(
                [t["log_probability"] for t in records],
                dtype=torch.float32,
            )

            log_probs, entropies, _ = evaluate_actions(
                model,
                batch_inputs([t["observations"] for t in records]),
                slot_actions,
                tile_actions,
                slot_masks,
                tile_masks,
            )

            ratio = (log_probs - old_log_probs).exp()
            adv = normalized_advantages[positions]

            clip = SETTINGS["clip_range"]

            policy_loss = -torch.minimum(
                ratio * adv,
                ratio.clamp(1 - clip, 1 + clip) * adv,
            ).mean()
            entropy_bonus = entropies.mean()

            # ----- Critic: uniform sample of ALL ticks -----
            critic_indices = torch.randint(
                len(transitions),
                size=(critic_batch_size,),
            )
            critic_inputs = batch_inputs([
                transitions[index]["observations"]
                for index in critic_indices.tolist()
            ])
            _, _, values = model(critic_inputs)

            value_loss = (
                values - returns[critic_indices]
            ).square().mean()

            loss = (
                policy_loss
                + SETTINGS["value_coefficient"] * value_loss
                - SETTINGS["entropy_coefficient"] * entropy_bonus
            )

            # Scale a short final actor minibatch proportionally.
            loss = loss * (len(records) / actor_batch_size)
            assert torch.isfinite(loss), "Non-finite PPO loss"

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                SETTINGS["max_grad_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            step_count += 1

            # Check the complete rollout's meaningful choices.
            current_kl = measure_policy_kl(model, transitions)

            print(
                f"Epoch {epoch + 1}, step {step_count}: "
                f"policy={policy_loss.item():.5f}, "
                f"value={value_loss.item():.5f}, "
                f"entropy={entropy_bonus.item():.4f}, "
                f"KL={current_kl:.6f}"
            )

            if current_kl > target_kl:
                print("Stopping PPO update: KL threshold exceeded.")
                return

def collect_match(
    model,
    seed,
    opponent_fn=opponent_policy,
    seat=0,
    opponent_deck=None,
):
    battle = cr_sim.Battle()

    if opponent_deck is None:
        opponent_deck = DECK

    decks = [None, None]
    decks[seat] = DECK
    decks[1 - seat] = opponent_deck

    battle.reset(seed=seed, deck0=decks[0], deck1=decks[1])

    my_view = battle.view(seat)
    opponent_view = battle.view(1 - seat)

    transitions = []

    action_counts = Counter()
    deployment_results = Counter()
    choice_ticks = 0
    total_ticks = 0

    while not battle.done():
        obs = my_view.observe()
        # Encodes observation into neural net friendly format
        model_input = encode_values(obs)
        slot_mask, tile_masks = action_masks(obs)

        slot_action, tile_id, log_probability, value_estimate = select_action(
            model, model_input, slot_mask, tile_masks
        )
        action = decode_action(slot_action, tile_id)
        total_ticks += 1
        if sum(slot_mask) > 1:
            choice_ticks += 1

        action_counts["wait" if slot_action == 0 else "deploy"] += 1

        # Choose opponents move and advance
        opponent_action = opponent_fn(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)
        actions = [None, None]
        actions[seat] = action
        actions[1 - seat] = opponent_action
        battle.step(*actions)

        if slot_action != 0:
            status = my_view.observe()["last_action"]
            deployment_results[status] += 1
        # Calculate reward and save experience
        done = battle.done()
        reward = 0.0
        if done:
            reward = {"win": 1, "draw": 0, "loss": -1}[my_view.result()]
        # Stores: state before action, selected action #, available actions for that move, feedback after move, state after move, whether step ended match
        transitions.append({
            "observations": model_input,
            "slot_action": slot_action,
            "tile_id": tile_id,
            "slot_mask": slot_mask,
            "tile_masks": tile_masks,
            "log_probability": log_probability,
            "value_estimate": value_estimate,
            "reward": reward,
            "done": done,
        })
        assert slot_mask[slot_action]

        if slot_action != 0:
            assert tile_id is not None
            assert tile_masks[slot_action - 1][tile_id]
    print("Deployment results:", dict(deployment_results))

    rejected = sum(
        count for status, count in deployment_results.items()
        if status != "ok"
    )
    print("Rejected deployments:", rejected)
    return transitions, my_view.result()


def main():
    # Create battle and player views; make battle with same deck for both sides
    battle = cr_sim.Battle()
    battle.reset(seed = 0, deck0 = DECK, deck1 = DECK)
    # Create and check network; encodes starting position and uses its length to determine networks input size
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimator",
        choices=("mc", "gae"),
        required=True,
    )
    parser.add_argument("--run-seed", type=int, default=12345)
    args = parser.parse_args()

    gae_lambda = 1.0 if args.estimator == "mc" else 0.9995

    CHECKPOINT_PATH = (
        Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "gandalf"
        / f"compare_{args.estimator}"
        / "latest.pt"
    )

    torch.manual_seed(args.run_seed)
    random.seed(args.run_seed)

    training_setup = {
        "estimator": args.estimator,
        "gae_lambda": gae_lambda,
        "gamma": SETTINGS["gamma"],
        "run_seed": args.run_seed,
        "target_choices": SETTINGS["target_choices"],
        "minimum_matches": SETTINGS["minimum_matches"],
    }

    experiment_metadata = {
        "schema_version": 2,
        "training_setup": dict(training_setup),
        "settings": dict(SETTINGS),
        "reward": "terminal_win_1_loss_minus1_draw_0",
        "opponent_policy": "random_agent.policy",
        "opponent_deck": list(DECK),
    }

    print("Training setup:", training_setup)
    model = ActorCritic(len(CARD_NAMES))
    # optimizer will adjust model's weights using the gradient we calculate
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"])
    # Converts python list into PyTorch tensor w/ 32-bit floating point #s
    # .no_grad ensures data for backpropogation not stored for now; calculates models predicted values for each output, and estimated reward
    
    config = {
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

    target_choices = SETTINGS["target_choices"]
    next_seed = 1
    completed_batches = 0
    # Restores learned weights and optimizers accumulated status
    saved_progress = None
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
        saved_progress = checkpoint.get("progress")
        if checkpoint["config"] != config:
            raise ValueError(
                "Checkpoint configuration differs from this model. "
                "Use a new checkpoint filename for this experiment."
            )
        saved_setup = checkpoint.get("training_setup")
        saved_metadata = checkpoint.get("experiment_metadata")

        if saved_metadata is not None:
            if saved_metadata != experiment_metadata:
                raise ValueError(
                    "Recorded experiment settings differ from this run."
                )
        else:
            print(
                "Legacy checkpoint: detailed metadata will be recorded "
                "from this continuation onward."
            )

        if saved_setup is not None:
            if saved_setup != training_setup:
                raise ValueError(
                    "Training settings differ from this experiment's checkpoint."
                )
        elif checkpoint["completed_batches"] != 150:
            raise ValueError(
                "Expected the original batch-150 checkpoint when starting a branch."
            )
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        assert all(
            group["lr"] == SETTINGS["learning_rate"]
            for group in optimizer.param_groups
        ), "Loaded optimizer learning rate differs from SETTINGS"
        next_seed = checkpoint["next_seed"]
        completed_batches = checkpoint["completed_batches"]

        print(f"Resuming after {completed_batches} batches")
    else:
        print("Starting a new model.")

    progress = (
        dict(saved_progress)
        if saved_progress is not None
        else {
            "counted_since_batch": completed_batches,
            "simulator_ticks": 0,
            "meaningful_choices": 0,
            "completed_matches": 0,
        }
    )

    initial_path = CHECKPOINT_PATH.parent / "initial.pt"

    if completed_batches == 0 and not initial_path.exists():
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "completed_batches": 0,
                "experiment_metadata": experiment_metadata,
                "progress": dict(progress),
            },
            initial_path,
        )
    target_batches = 250

    while completed_batches < target_batches:
        collection_seed = args.run_seed + 10_000 * completed_batches
        random.seed(collection_seed)
        torch.manual_seed(collection_seed)
        # Runs 3 times - store choices, results, and batch for traning
        batch = []
        choices = 0
        results = Counter()
        # Run matches until target number of choices collected
        while choices < target_choices or sum(results.values()) < SETTINGS["minimum_matches"]:
            transitions, result = collect_match(
                model,
                next_seed,
                seat=(next_seed - 1) % 2,
            )
            next_seed += 1
            # Store transitions of match in batch, and add to choices/results counters
            batch.extend(transitions)
            choices += sum(
                sum(t["slot_mask"]) > 1
                for t in transitions
            )
            results[result] += 1

        print(
            f"Batch {completed_batches + 1}: "
            f"{len(batch)} ticks, {choices} choices, "
            f"results={dict(results)}"
        )

        returns, advantages = calculate_returns_and_advantages(
            batch,
            gamma=SETTINGS["gamma"],
            gae_lambda=gae_lambda,
        )

        progress["simulator_ticks"] += len(batch)
        progress["meaningful_choices"] += choices
        progress["completed_matches"] += sum(results.values())

        print("Experience totals:", progress)
        report_rollout_quality(batch, returns, advantages)
        report_exploration(model, batch)
        ppo_update(model, optimizer, batch, returns, advantages)

        completed_batches += 1

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "next_seed": next_seed,
            "completed_batches": completed_batches,
            "training_setup": training_setup,
            "experiment_metadata": experiment_metadata,
            "progress": dict(progress),
        }

        temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(CHECKPOINT_PATH)
        if completed_batches in (175, 200, 225, 230):
            snapshot_path = CHECKPOINT_PATH.parent / f"gandalf_batch{completed_batches}.pt"
            torch.save(checkpoint, snapshot_path)
            print(f"Saved snapshot: {snapshot_path.name}")

        print(f"Saved checkpoint after batch {completed_batches}.")

# Ensures file is only run directly, not while being imported
if __name__ == "__main__":
    main()