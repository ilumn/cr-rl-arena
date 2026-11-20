# Create and reset a battle
# imports - bring in tensor, neural network operations, ActorCritic, battle simulator, opponent_policy
import torch
import random
from collections import Counter
from pathlib import Path
import cr_sim
from HAL9000 import (
    NUM_ACTIONS,
    DECK, 
    encode_observation,
    action_mask,
    decode_action,
    ENTITY_TYPES, 
    POSITIONS,
)
from hal_brains import ActorCritic
from random_agent import policy as opponent_policy

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "hal_baseline.pt"
)

def calculate_returns_and_advantages(transitions, gamma=0.9999):
    # return is reward action eventually leads to, with distant rewards discounted; e.g. loss would have final return -1, earlier returns less negative values
    # gamma controls discount
    returns = []
    future_return = 0.0

    for transition in reversed(transitions):
        if transition["done"]:
            future_return = 0.0

        future_return = (
            transition["reward"] + gamma * future_return
        )
        returns.append(future_return)

    returns.reverse()

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(
        [t["value_estimate"] for t in transitions],
        dtype=torch.float32
    )
    # compares calculated return with critic's prediction
    advantages = returns - values

    return returns, advantages

def ppo_update(model, optimizer, transitions, returns, advantages):
    # Convert collected batch into batches of tensors
    observations = torch.tensor(
        [t["observations"] for t in transitions],
        dtype=torch.float32
    )
    actions = torch.tensor(
        [t["action_id"] for t in transitions],
        dtype=torch.long
    )
    masks = torch.tensor(
        [t["action_mask"] for t in transitions],
        dtype=torch.bool
    )
    old_log_probs = torch.tensor(
        [t["log_probability"] for t in transitions],
        dtype=torch.float32
    )

    # Fix targets for this update
    returns = returns.detach()
    advantages = advantages.detach()

    choice_mask = masks.sum(dim=1) > 1

    if not choice_mask.any():
        print("Skipping update: no meaningful choices collected.")
        return
    # Standardize advantages to keep their scale manageable (gets z score)
    actor_advantages = advantages[choice_mask]
    actor_advantages = (
        (actor_advantages - actor_advantages.mean())
        / (actor_advantages.std(unbiased=False) + 1e-8)
    )

    for epoch in range(4):
        # Gradients are enabled, so these predictions will train model
        logits, values = model(observations)
        masked_logits = logits.masked_fill(~masks, float("-inf"))
        distribution = torch.distributions.Categorical(
            logits=masked_logits
        )

        new_log_probs = distribution.log_prob(actions)
        # Compare current action probabilities with collection-time ones
        ratio = torch.exp(
            new_log_probs[choice_mask] - old_log_probs[choice_mask]
        )
        with torch.no_grad():
            log_ratio = (
                new_log_probs[choice_mask] - old_log_probs[choice_mask]
            )

            approximate_kl = (
                (ratio - 1) - log_ratio
            ).mean().item()

            clip_fraction = (
                (ratio - 1).abs() > 0.2
            ).float().mean().item()
        # PPO limits incentive for large probability changes
        unclipped = ratio * actor_advantages
        clipped = torch.clamp(ratio, .8, 1.2) * actor_advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        # Teach critic to predict the observed returns
        value_loss = (values - returns).square().mean()
        # Encourage the policy to continue exploring
        entropy = distribution.entropy()[choice_mask].mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        assert torch.isfinite(loss), "Non-finite training loss"

        before = model.policy_head.weight.detach().clone()
        optimizer.zero_grad()
        # Calculates gradients for backpropogation
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), (
                    f"Non-finite gradient: {name}"
                )

        policy_gradient = model.policy_head.weight.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # Applies the changes to the weights
        optimizer.step()
        weight_change = (
            model.policy_head.weight.detach() - before
        ).norm().item()

        print(
            f"Policy gradient: {policy_gradient:.8f}, "
            f"policy weight change: {weight_change:.8f}"
            f"Update {epoch + 1}: "
            f"policy={policy_loss.item():.8f}, "
            f"value={value_loss.item():.4f}, "
            f"entropy={entropy.item():.4f}"
            f"Approx KL: {approximate_kl:.8f}, "
            f"clip fraction: {clip_fraction:.3f}"
        )

def collect_match(model, seed):
    # Create battle and player views; make battle with same deck for both sides
    battle = cr_sim.Battle()
    battle.reset(seed=seed, deck0=DECK, deck1=DECK)

    my_view = battle.view(0)
    opponent_view = battle.view(1)

    transitions = []

    action_counts = Counter()
    deployment_results = Counter()
    choice_entropies = []
    choice_ticks = 0
    total_ticks = 0

    while not battle.done():
        obs = my_view.observe()
        # Encodes observation into neural net friendly format
        model_input = encode_observation(obs)
        # Determines allowed action ids
        mask = action_mask(obs)
        # randomly chooses a valid action
        observation_tensor = torch.tensor(model_input, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            # Calculates models predicted moves and predicted value
            logits, value = model(observation_tensor)
            # use ~ to find which moves are illegal
            masked_logits = logits.masked_fill(~mask_tensor, float("-inf"))
            # Calculate probability distribution of all possible moves
            distribution = torch.distributions.Categorical(logits=masked_logits)
            # Choose a move based on calculated probabilities
            sampled_action = distribution.sample()
            action_id = sampled_action.item()

            total_ticks += 1

            if sum(mask) > 1:
                choice_ticks += 1
                choice_entropies.append(distribution.entropy().item())

            action_counts["wait" if action_id == 0 else "deploy"] += 1
            # Records how likely this action was under current policy
            log_probability = distribution.log_prob(sampled_action).item()
            # Records estimated future reward
            value_estimate = value.item()

        action = decode_action(action_id)

        # Choose opponents move and advance
        opponent_action = opponent_policy(opponent_view.observe())
        if opponent_action is not None:
            opponent_action = tuple(opponent_action)
        battle.step(action, opponent_action)

        if action_id != 0:
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
            "action_id": action_id,
            "action_mask": mask,
            "reward": reward,
            #"next_observation": encode_observation(my_view.observe()),
            "done": done,
            "log_probability": log_probability,
            "value_estimate": value_estimate,
        })
        assert mask[action_id], "Model selected an unavailable action"
    print(
        f"Deployment results: ok ({deployment_results["ok"]}) & not enough elixir ({deployment_results["not_enough_elixir"]})"
    )
    print(
        f"Hal could move in {choice_ticks} / {total_ticks} positions"
        f"({100 * choice_ticks / total_ticks:.2f}%)"
    )
    print(f"Hal attempted deployment in {action_counts["deploy"]} positions")
    if choice_entropies:
        print(f"Choice entropy mean: {sum(choice_entropies) / len(choice_entropies)}")
    return transitions, my_view.result()


def main():
    # Create battle and player views; make battle with same deck for both sides
    battle = cr_sim.Battle()
    battle.reset(seed = 0, deck0 = DECK, deck1 = DECK)
    # Create and check network; encodes starting position and uses its length to determine networks input size
    features = encode_observation(battle.view(0).observe())
    model = ActorCritic(len(features), NUM_ACTIONS)
    # optimizer will adjust model's weights using the gradient we calculate
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Converts python list into PyTorch tensor w/ 32-bit floating point #s
    # .no_grad ensures data for backpropogation not stored for now; calculates models predicted values for each output, and estimated reward
    
    config = {
        "encoding_version": 1,
        "input_size": len(features),
        "num_actions": NUM_ACTIONS,
        "deck": DECK,
        "entity_types": ENTITY_TYPES,
        "positions": POSITIONS, 
        "gamma": 0.9999,
    }

    target_choices = 32
    next_seed = 1
    completed_batches = 0
    # Restores learned weights and optimizers accumulated status
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["config"] != config:
            raise ValueError(
                "Checkpoint configuration differs from this model. "
                "Use a new checkpoint filename for this experiment."
            )
        
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        next_seed = checkpoint["next_seed"]
        completed_batches = checkpoint["completed_batches"]

        print(f"Resuming after {completed_batches} batches")
    else:
        print("Starting a new model.")

    target_batches = 50

    while completed_batches < target_batches:
        # Runs 3 times - store choices, results, and batch for traning
        batch = []
        choices = 0
        results = Counter()
        # Run matches until target number of choices collected
        while choices < target_choices:
            transitions, result = collect_match(model, next_seed)
            next_seed += 1
            # Store transitions of match in batch, and add to choices/results counters
            batch.extend(transitions)
            choices += sum(
                sum(t["action_mask"]) > 1 
                for t in transitions
            )
            results[result] += 1

        print(
            f"Batch {completed_batches + 1}: "
            f"{len(batch)} ticks, {choices} choices, "
            f"results={dict(results)}"
        )

        returns, advantages = calculate_returns_and_advantages(batch)
        ppo_update(model, optimizer, batch, returns, advantages)
        completed_batches += 1

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "next_seed": next_seed,
            "completed_batches": completed_batches,
        }

        temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(CHECKPOINT_PATH)
        if completed_batches in (20, 30, 50):
            snapshot_path = CHECKPOINT_PATH.parent / f"hal_batch{completed_batches}.pt"
            torch.save(checkpoint, snapshot_path)
            print(f"Saved snapshot: {snapshot_path.name}")

        print(f"Saved checkpoint after batch {completed_batches}.")

# Ensures file is only run directly, not while being imported
if __name__ == "__main__":
    main()