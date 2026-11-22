import json
from pathlib import Path

# A deck is required to join: 8 different playable cards, at most one champion.
REPO_ROOT = Path(__file__).resolve().parents[2]
CARDS_PATH = REPO_ROOT / "gym" / "data" / "cards.json"
DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "fireball", "minipekka"]
with CARDS_PATH.open() as file:
    card_definitions = json.load(file)["cards"]

ENTITY_TYPES = sorted(
    name
    for name, definition in card_definitions.items()
    if definition.get("type") in ("troop", "building")
)
# entitiy types represents the name all cards of type troop or building, plus unknown to deal with edge cases
ENTITY_TYPES.append("unknown")
# type index maps each name to its index in entity types, creating easy access
TYPE_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

BOARD_WIDTH = 18
BOARD_HEIGHT = 32
NUM_TILES = BOARD_WIDTH * BOARD_HEIGHT  # 576

# 0 = wait; 1–4 = play hand slot 0–3.
NUM_SLOT_ACTIONS = 5
MAX_ENTITIES = 64

# Include spells, summoned units, evolved variants, and towers.
CARD_NAMES = ["unknown"] + sorted(card_definitions)
CARD_INDEX = {name: index for index, name in enumerate(CARD_NAMES)}

PLACEMENT_MARGIN = 0.25  # Tiles; initial value to test.


def structure_boxes(obs):
    """Living structures, in the observing player's coordinates."""
    boxes = []

    for entity in obs["entities"]:
        (
            owner, card, x, y, hp, max_hp,
            flying, is_tower, deploying, uid, half_w, half_h,
        ) = entity

        definition = card_definitions.get(card, {})
        is_building = definition.get("type") == "building"

        if hp <= 0 or not (is_tower or is_building):
            continue

        # Towers expose box dimensions. Ordinary buildings may expose
        # zero half-extents, so conservatively bound their collision circle.
        if half_w == 0 and half_h == 0:
            radius = definition.get("collision_radius", 0.5)
            half_w = radius
            half_h = radius

        boxes.append((x, y, half_w, half_h))

    return boxes


def ground_spawn_clear(definition, x, y, boxes, friendly_towers):
    """Initial filter for the ordinary troops in Gandalf's current deck."""
    if (
        definition.get("type") != "troop"
        or definition.get("flying", False)
    ):
        return True

    count = definition.get("count", 1)

    if count == 1:
        offsets = [(0.0, 0.0)]
    elif count == 2:
        # Current engine's side-by-side pair: covers Archers.
        offsets = [(-0.35, 0.0), (0.35, 0.0)]
    else:
        raise ValueError(
            "Add spawn-formation handling before using this "
            f"ground troop group of {count}."
        )

    clearance = (
        definition.get("collision_radius", 0.5)
        + PLACEMENT_MARGIN
    )

    for offset_x, offset_y in offsets:
        spawn_x = x + offset_x
        spawn_y = y + offset_y
        # Avoid ground troops walking straight into a friendly tower.
        for center_x, center_y, half_w, half_h in friendly_towers:
            behind_tower = spawn_y < center_y - half_h
            in_tower_corridor = (
                abs(spawn_x - center_x) < half_w + clearance
            )

            if behind_tower and in_tower_corridor:
                return False
        for center_x, center_y, half_w, half_h in boxes:
            overlaps = (
                abs(spawn_x - center_x) < half_w + clearance
                and abs(spawn_y - center_y) < half_h + clearance
            )
            if overlaps:
                return False

    return True

def card_id(name):
    return CARD_INDEX.get(name, 0)

def encode_observation(obs):
    model_input = []
    model_input.append(obs["elixir"] / 10)
    # Time and crowns.
    model_input.append(obs["tick"] / 6000.0)
    model_input.extend(crown / 3.0 for crown in obs["crowns"])

    # The next card in your cycle.
    model_input.extend(
        float(obs["next_card"] == name)
        for name in DECK
    )
    # Stores list of binary numbers [0 0 1] representing which phase game is in
    model_input.extend([
        float(obs["phase"] == phase) for phase in ("single", "double", "overtime")
    ])
    # appends 32 entries - each 8 binary values [0 0 0 0 0 0 1 0] representing whether cards from deck are in our hand
    for card in obs["hand"]:
        model_input.extend([
            float(card == name) for name in DECK
        ])
    # store tower health with 6 entries - default 0 in case one is destroyed
    tower_health = [0.0] * 6
    # use obs["entities"] to get percent hp remaining for each tower, stored by index
    for entry in obs["entities"]:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entry

        if not is_tower:
            continue
        if card == "king_tower":
            tower_index = 2
        else:
            tower_index = 0 if x < 9 else 1

        index = owner * 3 + tower_index
        tower_health[index] = hp / max_hp if max_hp > 0 else 0.0
    model_input.extend(tower_health)
    # create a list representing how many of each troop type there are in certain locations; eight locations, - default 0
    counts = [
        [[0.0] * len(ENTITY_TYPES) for _ in range(8)]
        for _ in range(2)
    ]
    for entry in obs["entities"]:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entry

        if is_tower:
            continue
        # lane represents left/right with 0/1
        # band represents which fourth of the map to place with int 0-3; min max logic ensures it never goes beyond this range (32 / 8 divides into 4 even sections)
        lane = 0 if x < 9 else 1
        band = max(0, min(3, int(y / 8)))
        # multiplying by two and occasionally adding one guarantees all unique numbers from 0-7
        region = band * 2 + lane
        # use type index to get the index of the card in entity types
        troop_type = TYPE_INDEX.get(card, TYPE_INDEX["unknown"])
        # update counter in counts, which stores how many of whos troop type there are in which area 
        counts[owner][region][troop_type] += 1
    # add every count value to the list
    model_input.extend([
        count 
        for owner_counts in counts 
        for region_counts in owner_counts 
        for count in region_counts
    ])

    return model_input
    # Your king tower is at (9, 3) and your princess towers at (3.5, 6.5) and (14.5, 6.5). The opponent's are at (9, 29), (3.5, 25.5) and (14.5, 25.5).

def encode_values(obs):
    # Ignore dead entities.
    entities = [
        entity for entity in obs["entities"]
        if entity[4] > 0
    ]

    # ----- Tower health and opened pockets -----
    # Order: own left/right/king, enemy left/right/king.
    tower_health = [0.0] * 6
    enemy_princess_alive = [False, False]

    for entity in entities:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entity

        if not is_tower:
            continue

        if card == "king_tower":
            tower_slot = 2
        else:
            tower_slot = 0 if x < 9 else 1

            if owner == 1:
                enemy_princess_alive[tower_slot] = True

        tower_health[owner * 3 + tower_slot] = (
            hp / max_hp if max_hp > 0 else 0.0
        )

    pockets = [
        float(not alive) for alive in enemy_princess_alive
    ]

    # ----- Regional summaries: include ALL non-tower entities -----
    # Per region: ground count, ground HP, flying count, flying HP.
    regions = [
        [[0.0] * 4 for _ in range(8)]
        for _ in range(2)
    ]

    for entity in entities:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entity

        if is_tower:
            continue

        lane = 0 if x < 9 else 1
        band = max(0, min(3, int(y / 8)))
        region = band * 2 + lane
        offset = 2 if flying else 0

        # Fixed scaling; these values can exceed 1.
        regions[owner][region][offset] += 0.1
        regions[owner][region][offset + 1] += hp / 1000.0

    regional_features = [
        value
        for owner_regions in regions
        for region in owner_regions
        for value in region
    ]

    # ----- Select detailed entity records -----
    # Keep towers first. Then prioritize entities near opponents.
    # This is an initial overflow rule, not a tactical oracle.
    positions_by_owner = [
        [(e[2], e[3]) for e in entities if e[0] == owner]
        for owner in (0, 1)
    ]

    def priority(entity):
        owner, card, x, y = entity[:4]
        is_tower = entity[7]

        enemy_positions = positions_by_owner[1 - owner]
        distance = min(
            (
                (x - enemy_x) ** 2 + (y - enemy_y) ** 2
                for enemy_x, enemy_y in enemy_positions
            ),
            default=float("inf"),
        )

        return (
            0 if is_tower else 1,
            distance,
            entity[9],  # Stable entity ID breaks ties.
        )

    selected = sorted(entities, key=priority)[:MAX_ENTITIES]

    entity_ids = [0] * MAX_ENTITIES
    entity_features = [[0.0] * 9 for _ in range(MAX_ENTITIES)]
    entity_mask = [False] * MAX_ENTITIES

    for slot, entity in enumerate(selected):
        (
            owner, card, x, y, hp, max_hp,
            flying, is_tower, deploying, *_,
        ) = entity

        definition = card_definitions.get(card, {})

        entity_ids[slot] = card_id(card)
        entity_features[slot] = [
            float(owner),
            x / 18.0,
            y / 32.0,
            hp / max_hp if max_hp > 0 else 0.0,
            hp / 1000.0,
            float(flying),
            float(definition.get("type") == "building"),
            float(is_tower),
            float(deploying),
        ]
        entity_mask[slot] = True

    # ----- Hand and next card -----
    hand_ids = [card_id(card) for card in obs["hand"]]

    hand_features = [
        [cost / 10.0, float(affordable)]
        for cost, affordable in zip(
            obs["hand_cost"], obs["legal"]
        )
    ]

    next_card = obs["next_card"]
    next_cost = card_definitions.get(next_card, {}).get("elixir", 0)

    # ----- Global features -----
    global_features = [
        obs["elixir"] / 10.0,
        *[
            float(obs["phase"] == phase)
            for phase in ("single", "double", "overtime")
        ],
        obs["tick"] / 6000.0,
        *[crowns / 3.0 for crowns in obs["crowns"]],
        *tower_health,
        *pockets,
    ]

    return {
        "entity_ids": entity_ids,
        "entity_features": entity_features,
        "entity_mask": entity_mask,
        "regions": regional_features,
        "hand_ids": hand_ids,
        "hand_features": hand_features,
        "next_card_id": card_id(next_card),
        "next_card_cost": next_cost / 10.0,
        "globals": global_features,
    }

def decode_tile(tile_id):
    if not 0 <= tile_id < NUM_TILES:
        raise ValueError(f"Invalid tile ID: {tile_id}")

    row, column = divmod(tile_id, BOARD_WIDTH)
    return column + 0.5, row + 0.5


def decode_action(slot_action, tile_id=None):
    if not 0 <= slot_action < NUM_SLOT_ACTIONS:
        raise ValueError(f"Invalid slot action: {slot_action}")

    if slot_action == 0:
        return None

    if tile_id is None:
        raise ValueError("Playing a card requires a tile ID")

    hand_slot = slot_action - 1
    x, y = decode_tile(tile_id)

    return (hand_slot, x, y)

def action_masks(obs):
    boxes = structure_boxes(obs)
    friendly_towers = [
        (entity[2], entity[3], entity[10], entity[11])
        for entity in obs["entities"]
        if entity[0] == 0 and entity[7] and entity[4] > 0
    ]
    enemy_princess_alive = [False, False]

    for entity in obs["entities"]:
        owner, card, x, y, hp, max_hp, flying, is_tower, *_ = entity

        if (
            owner == 1
            and is_tower
            and card != "king_tower"
            and hp > 0
        ):
            lane = 0 if x < 9 else 1
            enemy_princess_alive[lane] = True

    slot_mask = [True]  # Waiting is always available.
    tile_masks = []

    for hand_slot, card in enumerate(obs["hand"]):
        definition = card_definitions[card]
        affordable = bool(obs["legal"][hand_slot])

        is_spell = definition.get("type") == "spell"
        deploy_anywhere = definition.get("deploy_anywhere", False)

        card_tile_mask = []

        for tile_id in range(NUM_TILES):
            x, y = decode_tile(tile_id)
            lane = 0 if x < 9 else 1

            if is_spell or deploy_anywhere:
                placement_allowed = True
            else:
                own_side = y < 15
                opened_lane = not enemy_princess_alive[lane]
                pocket = opened_lane and y < 21
                placement_allowed = own_side or pocket
            spawn_clear = ground_spawn_clear(
                definition, x, y, boxes, friendly_towers
            )

            card_tile_mask.append(
                affordable and placement_allowed and spawn_clear
            )
        tile_masks.append(card_tile_mask)
        slot_mask.append(any(card_tile_mask))

    return slot_mask, tile_masks