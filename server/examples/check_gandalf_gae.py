import torch

from train_gandalf import calculate_returns_and_advantages


transitions = [
    {"reward": 0.0, "value_estimate": 0.2, "done": False},
    {"reward": 1.0, "value_estimate": 0.4, "done": True},
    {"reward": -1.0, "value_estimate": 0.1, "done": True},
]

# With gamma=lambda=1, GAE targets equal full-match returns.
returns, advantages = calculate_returns_and_advantages(
    transitions, gamma=1.0, gae_lambda=1.0
)

assert torch.allclose(
    returns, torch.tensor([1.0, 1.0, -1.0])
)
assert torch.allclose(
    advantages, torch.tensor([0.8, 0.6, -1.1])
)

# With lambda=0, targets are one-step bootstrapped targets.
returns, _ = calculate_returns_and_advantages(
    transitions, gamma=1.0, gae_lambda=0.0
)
assert torch.allclose(
    returns, torch.tensor([0.4, 1.0, -1.0])
)

print("GAE checks passed.")