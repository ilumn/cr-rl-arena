import torch
from torch import nn

# Describes how model processes observation
# nn.Module allows PyTorch to track layers and trainable parameters
# Actor - scores possible actions (think node activations of final layer)
# Critic - estimates expected future reward
class ActorCritic(nn.Module):
    def __init__(self, input_size, num_actions):
        # Initializes PyTorch module
        super().__init__()
        # Builds processing layers
        self.shared = nn.Sequential(
            # Layer one - input is all input nodes, output is result of this layers' 128 nodes
            nn.Linear(input_size, 128),
            # ReLU - takes max(0, x); simple non-linear function
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # Policy head - returns 25 scores (node activations) called logits
        self.policy_head = nn.Linear(128, num_actions)
        # Value returns expected reward estimate - one value
        self.value_head = nn.Linear(128, 1)

    # Defines calculation performed by logits, value = model(observation_tensor)
    def forward(self, observation):
        # hidden - 128 intermediate features
        hidden = self.shared(observation)
        
        logits = self.policy_head(hidden)
        # .squeeze(-1) removes final dimension of size 1, turning value output into scalar tensor
        value = self.value_head(hidden).squeeze(-1)
        return logits, value