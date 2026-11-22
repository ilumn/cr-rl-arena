"""Opt-in coordinate placement architecture. Original battle model is unchanged."""

import torch

from gandalf import BOARD_WIDTH, BOARD_HEIGHT, NUM_TILES, decode_tile
from model_brains import ActorCritic


class CoordinateTileHead(torch.nn.Module):
    """Score each tile with shared weights conditioned on board/card context."""

    def __init__(self, hidden_size=32):
        super().__init__()
        self.context = torch.nn.Linear(128 + 18, hidden_size)
        self.position = torch.nn.Linear(2, hidden_size, bias=False)
        self.score = torch.nn.Linear(hidden_size, 1, bias=False)
        positions = torch.tensor([decode_tile(i) for i in range(NUM_TILES)])
        self.register_buffer(
            'coordinates', positions / torch.tensor([BOARD_WIDTH, BOARD_HEIGHT]) * 2 - 1)

    def forward(self, context):
        hidden = torch.tanh(
            self.context(context).unsqueeze(-2) + self.position(self.coordinates))
        return self.score(hidden).squeeze(-1)


class CoordinateActorCritic(ActorCritic):
    def __init__(self, num_card_types):
        super().__init__(num_card_types)
        self.tile_head = CoordinateTileHead()


class PlacementTemperature(torch.nn.Module):
    """Evaluation-only wrapper: change tile sampling, preserve card selection."""

    def __init__(self, model, temperature):
        super().__init__()
        if not 0 < temperature < float('inf'):
            raise ValueError('Placement temperature must be finite and positive')
        self.model = model
        self.temperature = temperature

    def forward(self, inputs):
        slots, tiles, values = self.model(inputs)
        return slots, tiles / self.temperature, values
