from copy import deepcopy

import cr_sim

from gandalf import (
    DECK, NUM_TILES,
    decode_tile, decode_action, action_masks,
)


def tile_id(column, row):
    return row * 18 + column


battle = cr_sim.Battle()
battle.reset(seed=42, deck0=DECK, deck1=DECK)
obs = deepcopy(battle.view(0).observe())

# Controlled hand: all four cards affordable for this mask test.
obs["hand"] = ["knight", "fireball", "archers", "giant"]
obs["legal"] = [True] * 4

assert decode_tile(0) == (0.5, 0.5)
assert decode_tile(575) == (17.5, 31.5)
assert decode_action(0) is None
assert decode_action(1, 0) == (0, 0.5, 0.5)

slots, tiles = action_masks(obs)
assert len(slots) == 5
assert len(tiles) == 4
assert all(len(mask) == NUM_TILES for mask in tiles)

# Knight: own side allowed, enemy side blocked.
assert not tiles[0][tile_id(3, 8)]  # Inside clearance zone
assert tiles[0][tile_id(3, 10)]     # Clear own-side placement
assert not tiles[0][tile_id(3, 18)]

# Fireball: enemy-side targeting allowed.
assert tiles[1][tile_id(3, 25)]

# Remove the enemy left princess tower.
obs["entities"] = [
    entity for entity in obs["entities"]
    if not (
        entity[0] == 1
        and entity[7]
        and entity[1] != "king_tower"
        and entity[2] < 9
    )
]

slots, tiles = action_masks(obs)
assert tiles[0][tile_id(3, 20)]       # Left pocket
assert not tiles[0][tile_id(14, 20)] # Right remains closed
assert not tiles[0][tile_id(3, 21)]  # Too deep

# Unaffordable card: neither slot nor tiles available.
obs["legal"][0] = False
slots, tiles = action_masks(obs)
assert not slots[1]
assert not any(tiles[0])
assert slots[0]  # Waiting remains available.

print("Gandalf action checks passed.")

battle = cr_sim.Battle()
battle.reset(seed=42, deck0=DECK, deck1=DECK)
obs = deepcopy(battle.view(0).observe())

obs["hand"] = ["knight", "archers", "minions", "fireball"]
obs["legal"] = [True] * 4

slots, tiles = action_masks(obs)

# Own left princess tower is centered at (3.5, 6.5).
tower_tile = tile_id(3, 6)

assert not tiles[0][tower_tile]  # Knight blocked
assert not tiles[1][tower_tile]  # Archers blocked

# This ground-structure filter does not restrict flyers or spells.
assert tiles[2][tower_tile]
assert tiles[3][tower_tile]

# Legitimate positions behind and ahead of the tower remain available.
# Directly behind the left princess tower: now blocked.
assert not tiles[0][tile_id(3, 2)]

# Back of the arena, outside the tower's corridor: still available.
assert tiles[0][tile_id(0, 2)]

# Ahead of the tower: still available.
assert tiles[0][tile_id(3, 10)]

# Directly behind the king tower: blocked.
assert not tiles[0][tile_id(8, 0)]
assert tiles[0][tile_id(3, 10)]  # (3.5, 10.5), ahead
assert not tiles[1][tile_id(3, 2)]  # Archers blocked
assert tiles[2][tile_id(3, 2)]      # Minions unaffected
assert tiles[3][tile_id(3, 2)]      # Fireball unaffected

print("Structure-clearance checks passed.")