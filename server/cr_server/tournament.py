"""A single-elimination tournament, run by hand.

Agents enrol themselves by connecting; nothing is played until an organiser
says so. The organiser draws a bracket from whoever is in the pool, then starts
each pairing's games one at a time. This module is the bracket and its file on
disk; `server.py` owns the sockets and the games.

Phases
    lobby     collecting entrants; no bracket yet
    running   a bracket exists and has pairings left to play
    done      the final is decided

A pairing is decided by the first player to win `best_of // 2 + 1` games. A drawn
game counts for nobody, so the pairing simply needs another game. Byes go to the
top seeds and are settled as soon as the bracket is drawn.

`random_from` is the game of a pairing from which the players are dealt random
decks instead of their own. Each is dealt a different one, until the last game
of the series, which is a mirror: both get the same random deck. So a best of 5
with `random_from` 3 is two games on their own decks, two on different random
decks, and a mirrored decider. None means every game is on the players' own.
"""
from __future__ import annotations

import json
import os
import time

BEST_OF = (1, 3, 5)
SEEDING = ("rating", "random")
MAX_ENTRANTS = 64


def seed_order(size: int) -> list[int]:
    """Seeds in bracket order for a field of `size` (a power of two), so that the
    top seeds can only meet late: 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        n = len(order) * 2
        order = [s for a in order for s in (a, n + 1 - a)]
    return order


def round_name(left: int) -> str:
    """Name of a round with `left` rounds still to play after it."""
    return {0: "Final", 1: "Semifinals", 2: "Quarterfinals"}.get(left, f"Round of {2 ** (left + 1)}")


class Tournament:
    def __init__(self, path: str, auto: bool = False):
        self.path = path
        self.d = {"phase": "lobby", "auto": bool(auto), "excluded": [], "best_of": 1, "seeding": "rating",
                  "random_from": None, "rounds": [], "champion": None, "drawn_at": None}
        try:
            with open(path) as f:
                saved = json.load(f)
            if isinstance(saved, dict) and saved.get("phase") in ("lobby", "running", "done"):
                self.d.update(saved)
        except (OSError, ValueError):
            pass
        for p in self.pairings():
            p["live"] = False           # nothing survives a restart mid-game

    # ---- reading -----------------------------------------------------------
    @property
    def phase(self) -> str:
        return self.d["phase"]

    @property
    def auto(self) -> bool:
        return self.d["auto"]

    def pairings(self):
        return [p for r in self.d["rounds"] for p in r]

    def pairing(self, pid):
        return next((p for p in self.pairings() if p["id"] == pid), None)

    def entrants(self) -> set[str]:
        return {n for p in self.pairings() for n in p["players"] if n}

    def in_play(self, name: str) -> bool:
        """Whether `name` still has a pairing to play in the bracket."""
        return self.phase == "running" and any(
            name in p["players"] and p["winner"] is None for p in self.pairings())

    def need(self) -> int:
        return self.d["best_of"] // 2 + 1

    def dealt(self, p) -> str | None:
        """How the decks for the next game of pairing `p` are chosen: None (the
        players' own), "random" (a different random deck each) or "mirror" (the
        same random deck for both: the last game of the series, and any after it)."""
        start, n = self.d.get("random_from"), len(p["games"]) + 1
        if start is None or n < start:
            return None
        return "mirror" if n >= self.d["best_of"] else "random"

    # ---- changing ----------------------------------------------------------
    def save(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.d, f, separators=(",", ":"))
            os.replace(tmp, self.path)
        except OSError:
            pass                        # the bracket still lives in memory

    def set_auto(self, on: bool):
        self.d["auto"] = bool(on)
        self.save()

    def exclude(self, name: str, on: bool):
        ex = set(self.d["excluded"])
        (ex.add if on else ex.discard)(name)
        self.d["excluded"] = sorted(ex)
        self.save()

    def draw(self, entrants: list[str], best_of: int, seeding: str, random_from: int | None = None):
        """Build the bracket. `entrants` are in seed order, best first."""
        if self.phase != "lobby":
            raise ValueError("a tournament is already under way")
        if best_of not in BEST_OF or seeding not in SEEDING:
            raise ValueError("unknown format")
        if random_from is not None and (type(random_from) is not int or not 1 <= random_from <= best_of):
            raise ValueError("random decks must start at a game of the series")
        if len(set(entrants)) != len(entrants):
            raise ValueError("two entrants share a name")
        if not 2 <= len(entrants) <= MAX_ENTRANTS:
            raise ValueError(f"a bracket needs between 2 and {MAX_ENTRANTS} entrants")
        size = 2
        while size < len(entrants):
            size *= 2
        order = seed_order(size)
        rounds, n, r = [], size // 2, 0
        while n >= 1:
            rounds.append([{"id": f"{r}-{i}", "round": r, "slot": i, "players": [None, None],
                            "seeds": [None, None], "wins": [0, 0], "games": [], "winner": None,
                            "how": None, "live": False} for i in range(n)])
            n //= 2
            r += 1
        for i, p in enumerate(rounds[0]):
            for side in (0, 1):
                seed = order[2 * i + side]
                if seed <= len(entrants):
                    p["players"][side], p["seeds"][side] = entrants[seed - 1], seed
        self.d.update(phase="running", best_of=best_of, seeding=seeding, random_from=random_from, rounds=rounds,
                      champion=None, drawn_at=int(time.time() * 1000))
        for p in rounds[0]:
            if None in p["players"]:
                self._decide(p, 0 if p["players"][0] else 1, "bye")
        self.save()

    def _decide(self, p, side: int, how: str):
        p["winner"], p["how"], p["live"] = side, how, False
        rounds = self.d["rounds"]
        if p["round"] + 1 < len(rounds):
            nxt = rounds[p["round"] + 1][p["slot"] // 2]
            nxt["players"][p["slot"] % 2] = p["players"][side]
            nxt["seeds"][p["slot"] % 2] = p["seeds"][side]
        else:
            self.d["champion"] = p["players"][side]
            self.d["phase"] = "done"

    def playable(self, p) -> str | None:
        """Why a game can't be started in pairing `p`, or None if one can."""
        if p is None:
            return "no such pairing"
        if p["winner"] is not None:
            return "that pairing is already decided"
        if None in p["players"]:
            return "that pairing is still waiting for a player"
        if p["live"]:
            return "that pairing has a game in progress"
        return None

    def record(self, pid: str, winner: str | None, game: dict):
        """A game in pairing `pid` finished. `winner` is a player's name, or None for a draw."""
        p = self.pairing(pid)
        p["live"] = False
        side = p["players"].index(winner) if winner in p["players"] else None
        p["games"].append({**game, "winner": side})
        if side is not None:
            p["wins"][side] += 1
            if p["wins"][side] >= self.need():
                self._decide(p, side, "played")
        self.save()

    def walkover(self, pid: str, side: int):
        p = self.pairing(pid)
        problem = self.playable(p)
        if problem:
            raise ValueError(problem)
        if side not in (0, 1):
            raise ValueError("choose which player goes through")
        self._decide(p, side, "walkover")
        self.save()

    def reset(self):
        if any(p["live"] for p in self.pairings()):
            raise ValueError("a game is still being played")
        self.d.update(phase="lobby", rounds=[], champion=None, drawn_at=None)
        self.save()

    # ---- for the page ------------------------------------------------------
    def snapshot(self) -> dict:
        n = len(self.d["rounds"])
        return {
            "phase": self.phase, "auto": self.auto, "best_of": self.d["best_of"], "seeding": self.d["seeding"],
            "random_from": self.d.get("random_from"),
            "need": self.need(), "champion": self.d["champion"], "drawn_at": self.d["drawn_at"],
            "rounds": [{"name": round_name(n - 1 - i), "pairings": r} for i, r in enumerate(self.d["rounds"])],
        }
