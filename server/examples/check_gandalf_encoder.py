import math
from copy import deepcopy

import cr_sim

from gandalf import (
    DECK, MAX_ENTITIES, CARD_NAMES,
    card_id, encode_values,
)


def check_structure(encoded):
    """Check fixed shapes, valid IDs, finite numbers, and padding."""
    assert len(encoded["entity_ids"]) == MAX_ENTITIES
    assert len(encoded["entity_features"]) == MAX_ENTITIES
    assert all(len(row) == 9 for row in encoded["entity_features"])
    assert len(encoded["entity_mask"]) == MAX_ENTITIES
    assert len(encoded["regions"]) == 64
    assert len(encoded["hand_ids"]) == 4
    assert len(encoded["hand_features"]) == 4
    assert all(len(row) == 2 for row in encoded["hand_features"])
    assert len(encoded["globals"]) == 15

    ids = (
        encoded["entity_ids"]
        + encoded["hand_ids"]
        + [encoded["next_card_id"]]
    )
    assert all(
        isinstance(value, int) and 0 <= value < len(CARD_NAMES)
        for value in ids
    )

    def check_finite(value):
        if isinstance(value, list):
            for item in value:
                check_finite(item)
        else:
            assert math.isfinite(value), f"Invalid value: {value}"

    for value in encoded.values():
        check_finite(value)

    for index, occupied in enumerate(encoded["entity_mask"]):
        if not occupied:
            assert encoded["entity_ids"][index] == 0
            assert encoded["entity_features"][index] == [0.0] * 9


def troop(owner=1, x=3.5, y=18.5, hp=1000, uid=999):
    # Same entity layout as the simulator.
    return [
        owner, "knight", x, y, hp, 2000,
        False, False, False, uid, 0.0, 0.0,
    ]


def regional_index(owner, region, feature):
    # Features: ground count, ground HP, flying count, flying HP.
    return owner * 32 + region * 4 + feature


def main():
    battle = cr_sim.Battle()
    battle.reset(seed=42, deck0=DECK, deck1=DECK)

    base = deepcopy(battle.view(0).observe())
    base["entities"] = [list(e) for e in base["entities"]]

    # 1. Real initial observations from both seats.
    for seat in (0, 1):
        encoded = encode_values(battle.view(seat).observe())
        check_structure(encoded)
        assert sum(encoded["entity_mask"]) == 6
        assert encoded["globals"][-2:] == [0.0, 0.0]

    # 2. Damage changes HP, but not troop count.
    obs = deepcopy(base)
    obs["entities"].append(troop())
    healthy = encode_values(obs)

    obs["entities"][-1][4] = 500
    damaged = encode_values(obs)

    # x=3.5, y=18.5 means left lane, band 2: region 4.
    count_index = regional_index(1, 4, 0)
    hp_index = regional_index(1, 4, 1)

    assert healthy["regions"][count_index] == 0.1
    assert damaged["regions"][count_index] == 0.1
    assert healthy["regions"][hp_index] == 1.0
    assert damaged["regions"][hp_index] == 0.5

    slot = damaged["entity_ids"].index(card_id("knight"))
    record = damaged["entity_features"][slot]
    assert record[0] == 1.0             # Enemy
    assert math.isclose(record[1], 3.5 / 18)
    assert math.isclose(record[2], 18.5 / 32)
    assert record[3] == 0.25            # 500 / 2000 HP
    assert record[4] == 0.5             # 500 / 1000 HP

    # 3. Moving right transfers the regional features.
    obs["entities"][-1][2] = 14.5
    moved = encode_values(obs)

    assert moved["regions"][count_index] == 0.0
    assert moved["regions"][hp_index] == 0.0
    assert moved["regions"][regional_index(1, 5, 0)] == 0.1
    assert moved["regions"][regional_index(1, 5, 1)] == 0.5

    # 4. Friendly troops use separate regional channels.
    obs["entities"][-1][0] = 0
    friendly = encode_values(obs)
    assert friendly["regions"][regional_index(1, 5, 0)] == 0.0
    assert friendly["regions"][regional_index(0, 5, 0)] == 0.1

    # 5. Removing the enemy left tower opens only its pocket.
    obs = deepcopy(base)
    obs["entities"] = [
        e for e in obs["entities"]
        if not (
            e[0] == 1 and e[7]
            and e[1] != "king_tower" and e[2] < 9
        )
    ]
    encoded = encode_values(obs)
    assert encoded["globals"][-2:] == [1.0, 0.0]
    assert encoded["globals"][10] == 0.0  # Enemy left tower HP

    # 6. Overflow keeps towers and summarizes every troop.
    obs = deepcopy(base)
    obs["entities"].extend(
        troop(uid=1000 + i) for i in range(80)
    )
    encoded = encode_values(obs)
    check_structure(encoded)

    assert sum(encoded["entity_mask"]) == MAX_ENTITIES
    assert sum(row[7] for row in encoded["entity_features"]) == 6
    assert math.isclose(encoded["regions"][count_index], 8.0)
    assert encoded["regions"][hp_index] == 80.0

    # 7. Hand features and next-card identity.
    encoded = encode_values(base)
    assert encoded["hand_ids"] == [
        card_id(name) for name in base["hand"]
    ]
    assert encoded["hand_features"] == [
        [cost / 10.0, float(legal)]
        for cost, legal in zip(base["hand_cost"], base["legal"])
    ]
    assert encoded["next_card_id"] == card_id(base["next_card"])

    # 8. No entities: shape remains valid and all slots are padding.
    obs = deepcopy(base)
    obs["entities"] = []
    encoded = encode_values(obs)
    check_structure(encoded)
    assert not any(encoded["entity_mask"])

    print("All Gandalf encoder checks passed.")


if __name__ == "__main__":
    main()