"""Tournament server: continuous matches between agents, ranked by Elo.

Agents connect over WebSocket, join the queue, and are paired for
server-authoritative matches. The server owns the only real `cr_sim.Battle`;
agents just answer observations with actions. Results update a persistent Elo
leaderboard, and each match's (seed, action-log) is saved so it can be replayed.

The same port also serves a web page: the live standings at `/` and a replay
viewer for any saved match (see `web.py`).

Run:
    pip install cr-sim websockets        # cr-sim = the gym package
    python -m cr_server.server
Env:
    CR_ARENA_BIND        host:port         (default 0.0.0.0:8765)
    CR_ARENA_DATA        data dir          (default ./arena_data)
    CR_MOVE_TIMEOUT      seconds an agent has to answer each tick (default 2.0)

Protocol (JSON text frames)
  C->S  {"t":"join","name":str,"deck":[8 card names]}   (deck required; an
        illegal deck is refused with {"t":"error","msg":...} and the socket closed)
        {"t":"leaderboard"}                      (query then close)
  S->C  {"t":"joined","name","rating"}
        {"t":"start","opponent"}
        {"t":"obs","obs":{...}}                   (every tick; reply required)
  C->S  {"t":"act","tick":obs.tick,"action":[slot,x,y]|null}
  S->C  {"t":"result","result":"win"|"loss"|"draw","crowns":[mine,theirs],
         "rating","delta","match","seed"}
        {"t":"aborted","reason"}                  (match void: the opponent stopped answering)
        {"t":"leaderboard","standings":[...]}

Every agent plays as the bottom player: `obs` and `action` are in the agent's own
seat frame (see cr_sim), so an agent is never told, and never needs, its seat. The
seed is only sent with the result: before the match it would let an agent work out
the opponent's hand order.

Turns. An agent is asked for a move on every tick and has CR_MOVE_TIMEOUT seconds
to answer. Its reply must carry the `tick` it answers, so a reply that arrives late
is recognised and dropped instead of being taken for the next tick's move. No
answer in time means no move for that agent on that tick. An agent that lets
`match.MAX_MISSED_TURNS` turns in a row go by is disconnected, and that match is
closed without being recorded.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import suppress

import cr_sim
import websockets

from . import match
from .matchlog import UPSET_GAP, MatchLog
from .leaderboard import Leaderboard
from .web import Web

BIND = os.environ.get("CR_ARENA_BIND", "0.0.0.0:8765")
DATA_DIR = os.environ.get("CR_ARENA_DATA", "./arena_data")
MOVE_TIMEOUT = float(os.environ.get("CR_MOVE_TIMEOUT", "2.0"))
HUMAN_DECK = ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "fireball", "arrows"]


class WsAgent:
    """Wraps a connected websocket as a match-runner agent."""

    human = False

    def __init__(self, ws, name, deck):
        self.ws = ws
        self.name = name
        self.deck = deck
        self.alive = True
        self.queue_id = uuid.uuid4().hex

    async def send(self, obj):
        try:
            await self.ws.send(json.dumps(obj))
        except websockets.ConnectionClosed:
            self.alive = False

    async def act(self, obs, timeout):
        """Send the observation for this tick and wait for the reply to it.

        Returns (action, problem): problem is None for a usable reply, "missed"
        if none came in time, "malformed" if it could not be read. Replies tagged
        with any other tick are leftovers from a turn this agent already missed;
        they are dropped and the wait goes on, so an agent can never slip a step
        behind.
        """
        tick = obs["tick"]
        loop = asyncio.get_running_loop()
        try:
            await self.ws.send(json.dumps({"t": "obs", "obs": obs}))
            deadline = loop.time() + timeout
            while True:
                left = deadline - loop.time()
                if left <= 0:
                    return None, "missed"
                raw = await asyncio.wait_for(self.ws.recv(), left)
                try:
                    msg = json.loads(raw)
                except ValueError:
                    return None, "malformed"
                if not isinstance(msg, dict) or msg.get("t") != "act":
                    return None, "malformed"
                at = msg.get("tick")
                if not isinstance(at, int) or isinstance(at, bool):
                    return None, "malformed"      # every reply must say which tick it answers
                if at != tick:
                    continue                      # a late answer to another tick
                return msg.get("action"), None
        except websockets.ConnectionClosed:
            self.alive = False
            return None, None
        except asyncio.TimeoutError:
            return None, "missed"

    async def drop(self, why):
        """Tell the agent why it is being removed, then close its connection."""
        await self.send({"t": "error", "msg": why})
        self.alive = False
        try:
            await self.ws.close()
        except Exception:
            pass


class HumanAgent(WsAgent):
    """Sample clicks without waiting for a human response on every tick."""

    human = True

    def __init__(self, ws, name, deck):
        super().__init__(ws, name, deck)
        self.pending = None
        self.tick = -1

    async def receive(self):
        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict) or msg.get("t") != "act":
                    continue
                tick = msg.get("tick")
                if type(tick) is int and 0 <= self.tick - tick <= cr_sim.TICKS_PER_SECOND * 2:
                    action = match._validate(msg.get("action"))
                    if action is not None and (0 <= action[0] < 4 or action[0] == cr_sim.ABILITY_SLOT):
                        self.pending = (action, msg.get("card"), tick)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.alive = False

    async def act(self, obs, timeout):
        self.tick = obs["tick"]
        await self.send({"t": "obs", "obs": obs})
        pending, self.pending = self.pending, None
        if pending:
            action, card, tick = pending
            if self.tick - tick <= cr_sim.TICKS_PER_SECOND * 2:
                # A second click must not accidentally play the replacement card
                # after this hand slot has cycled.
                if action[0] == cr_sim.ABILITY_SLOT or card == obs["hand"][action[0]]:
                    return action, None
        return None, None


class Arena:
    def __init__(self):
        self.board = Leaderboard(os.path.join(DATA_DIR, "leaderboard.json"))
        self.idle: dict[str, WsAgent] = {}      # all waiting bots remain visible and claimable
        self._queue_changed = asyncio.Event()
        self.replays_dir = os.path.join(DATA_DIR, "replays")
        os.makedirs(self.replays_dir, exist_ok=True)
        self.log = MatchLog(DATA_DIR)           # match history behind the web page
        self.online: dict[str, int] = {}        # name -> open connections under it
        self.live = 0                           # matches in progress
        self._deck_checker = cr_sim.Battle()  # only used to validate decks at join

    async def handler(self, ws):
        try:
            raw = await asyncio.wait_for(ws.recv(), 30)
            msg = json.loads(raw)
        except Exception:
            return
        if not isinstance(msg, dict):
            await ws.send(json.dumps({"t": "error", "msg": "expected a JSON object"}))
            return
        if msg.get("t") == "play":
            await self._human_game(ws, msg)
            return
        if msg.get("t") == "leaderboard":
            await ws.send(json.dumps({"t": "leaderboard", "standings": self.board.standings()}))
            return
        if msg.get("t") != "join":
            await ws.send(json.dumps({"t": "error", "msg": "expected join or leaderboard"}))
            return

        name = str(msg.get("name") or f"anon-{random.randint(1000,9999)}")
        # A deck is required and is validated here, at join, so a missing or illegal
        # deck is refused to the agent that sent it instead of failing later inside
        # a match (which used to void the game for its opponent as well).
        deck = msg.get("deck")
        problem = self._deck_problem(deck)
        if problem:
            await ws.send(json.dumps({"t": "error", "msg": problem}))
            return
        agent = WsAgent(ws, name, deck)
        await agent.send({"t": "joined", "name": name,
                          "rating": round(self.board.get(name).rating, 1)})
        self._enqueue(agent)
        self.online[name] = self.online.get(name, 0) + 1
        try:
            await ws.wait_closed()
        finally:
            agent.alive = False
            self.idle.pop(agent.queue_id, None)
            left = self.online.get(name, 1) - 1
            if left > 0:
                self.online[name] = left
            else:
                self.online.pop(name, None)

    def standings_snapshot(self) -> dict:
        """Data for the web page: standings, recent results, live counts."""
        return self.log.snapshot(self.board.standings(), set(self.online), self.live)

    def matchmaking_snapshot(self) -> dict:
        return {
            "waiting": [{"id": ag.queue_id, "name": ag.name,
                         "rating": round(self.board.get(ag.name).rating, 1), "deck": ag.deck}
                        for ag in self.idle.values() if ag.alive],
            "live": self.live,
            "cards": self._deck_checker.deployable_cards(),
            "default_deck": HUMAN_DECK,
        }

    async def _human_game(self, ws, msg):
        deck = msg.get("deck", HUMAN_DECK)
        problem = self._deck_problem(deck)
        if problem:
            await ws.send(json.dumps({"t": "error", "msg": problem}))
            return
        bot_id = msg.get("bot")
        bot = self.idle.pop(bot_id, None) if isinstance(bot_id, str) else None
        if bot is None or not bot.alive:
            await ws.send(json.dumps({"t": "error", "msg": "That bot is no longer waiting. Please choose another."}))
            return
        # Claiming the bot and removing it from the ranked queue is one operation
        # with no await: two humans or the matchmaker cannot claim it together.
        name = "Human · " + (str(msg.get("name") or "Player").strip() or "Player")[:40]
        human = HumanAgent(ws, name, deck)
        reader = asyncio.create_task(human.receive())
        try:
            await self._play(human, bot)
        finally:
            human.alive = False
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader

    def _enqueue(self, agent):
        if agent.alive and not agent.human:
            self.idle[agent.queue_id] = agent
            self._queue_changed.set()

    def _deck_problem(self, deck):
        """Why `deck` can't be played, or None if it is legal."""
        if deck is None:
            return "a deck is required: send 8 card names"
        if not isinstance(deck, list) or not all(isinstance(c, str) for c in deck):
            return "deck must be a list of 8 card names"
        try:
            self._deck_checker.check_deck(deck)  # same rules Battle.reset enforces
        except ValueError as e:
            return f"invalid deck: {e}"
        return None

    async def matchmaker(self):
        while True:
            await self._queue_changed.wait()
            self._queue_changed.clear()
            waiting = [ag for ag in self.idle.values() if ag.alive]
            for i in range(0, len(waiting) - 1, 2):
                a, b = waiting[i:i + 2]
                self.idle.pop(a.queue_id, None)
                self.idle.pop(b.queue_id, None)
                asyncio.create_task(self._play(a, b))

    async def _play(self, a: WsAgent, b: WsAgent):
        rated = not (a.human or b.human)
        # Seats are drawn at random for every match: `a` is player 0 (bottom) from
        # here on. Queue order used to decide it, so with two agents connected the
        # same one was player 0 in every game.
        if random.random() < 0.5:
            a, b = b, a
        self.live += 1
        try:
            seed = random.getrandbits(63)
            for ag, opponent in ((a, b), (b, a)):
                start = {"t": "start", "opponent": opponent.name}
                if not rated:
                    start.update(rated=False, realtime=True)
                if ag.human:
                    start["arena"] = {
                        "w": cr_sim.WIDTH, "h": cr_sim.HEIGHT, "tps": cr_sim.TICKS_PER_SECOND,
                        "river_y": cr_sim.RIVER_Y, "river_half": cr_sim.RIVER_HALF,
                        "bridges": list(cr_sim.BRIDGES), "bridge_half": cr_sim.BRIDGE_HALF,
                        "regulation": cr_sim.REGULATION_TICKS,
                    }
                await ag.send(start)

            options = {"realtime": True} if not rated else {}
            res = await match.run_match(cr_sim, a, b, seed=seed, move_timeout=MOVE_TIMEOUT, **options)
            if res.get("aborted") is not None:
                # One or both agents stopped answering. The match is void: no rating
                # change, no result, no replay. The silent agent is disconnected so
                # it cannot stall the next opponent too; the other goes back in the queue.
                gone = [(a, b)[i] for i in res["aborted"]]
                for ag in (a, b):
                    if ag in gone:
                        await ag.drop(f"removed: no reply to {match.MAX_MISSED_TURNS} turns in a row")
                    elif rated:
                        await ag.send({"t": "aborted", "reason": "opponent stopped responding"})
                if rated:
                    print(f"[match void] {a.name} vs {b.name}: "
                          f"{', '.join(g.name for g in gone)} stopped responding at tick {res['ticks']}")
                    return
            out = res["outcome"]
            # forfeit: a disconnected agent loses to a connected one
            forfeit = False
            if not a.alive and b.alive:
                out, forfeit = 2, True
            elif not b.alive and a.alive:
                out, forfeit = 1, True
            elif not rated and not a.alive and not b.alive:
                out, forfeit = 3, True

            score_a = 1.0 if out == 1 else 0.0 if out == 2 else 0.5
            before = tuple(None if ag.human else self.board.get(ag.name).rating for ag in (a, b))
            delta = self.board.report(a.name, b.name, score_a) if rated else 0.0
            mid = self._record(a, b, res, out, forfeit, delta, before, rated=rated)

            # each agent hears the result from its own side, never as a seat number
            c = res["crowns"]
            for ag, won, mine, d in ((a, 1, c, delta), (b, 2, c[::-1], -delta)):
                await ag.send({"t": "result",
                               "result": "win" if out == won else "loss" if out in (1, 2) else "draw",
                               "crowns": mine, "rating": self._rating(ag),
                               "delta": round(d, 1), "match": mid, "seed": seed,
                               **({"rated": False} if not rated else {})})
            print(f"[match] {a.name} vs {b.name} -> {['--','P0','P1','draw'][out]} "
                  f"{res['crowns']} in {res['ticks']} ticks")
        except Exception as e:  # never let one match kill the server
            print(f"[match error] {a.name} vs {b.name}: {e!r}")
            if not rated:
                for ag in (a, b):
                    await ag.send({"t": "error", "msg": "The game could not finish. Please try again."})
        finally:
            self.live -= 1
            for x in (a, b):
                self._enqueue(x)

    def _rating(self, agent):
        return None if agent.human else round(self.board.get(agent.name).rating, 1)

    def _record(self, a, b, res, out, forfeit, delta, before, *, rated=True):
        """Save the replay file and append the match to the log the web page reads.

        `outcome` is the result that counted for the ratings; `sim_outcome` is what
        the simulator itself produced. They differ only for a forfeit, and the
        replay viewer checks a re-run against `sim_outcome`.
        """
        mid = self.log.next_id()
        fname = f"{mid:08d}.json"          # never built from agent names
        at = int(time.time() * 1000)
        r0, r1 = self._rating(a), self._rating(b)
        winner_before, loser_before = (before if out == 1 else before[::-1])
        saved = True
        try:
            with open(os.path.join(self.replays_dir, fname), "w") as f:
                json.dump({"id": mid, "at": at, "p0": a.name, "p1": b.name,
                           "deck0": a.deck, "deck1": b.deck,
                           **res, "outcome": out, "sim_outcome": res["outcome"],
                           "forfeit": forfeit, "rated": rated,
                           "human": [a.human, b.human]}, f)
        except OSError:
            saved = False
        try:
            self.log.add({
                "id": mid, "at": at, "p0": a.name, "p1": b.name, "outcome": out,
                "crowns": res["crowns"], "ticks": res["ticks"],
                "delta": round(delta, 1), "r0": r0, "r1": r1,
                "upset": rated and out in (1, 2) and not forfeit and winner_before < loser_before - UPSET_GAP,
                "rated": rated, "human": [a.human, b.human],
                "forfeit": forfeit, "deck0": a.deck, "deck1": b.deck,
                "moves": res.get("moves"), "ignored": res.get("ignored"),
                "file": fname if saved else None,
            })
        except OSError:
            pass
        return mid


async def main():
    host, port = BIND.rsplit(":", 1)
    arena = Arena()
    asyncio.create_task(arena.matchmaker())
    print(f"cr-arena listening on ws://{host}:{port}  (data: {DATA_DIR}, move timeout {MOVE_TIMEOUT}s)")
    web = Web(arena)
    async with websockets.serve(arena.handler, host, int(port), ping_interval=20,
                                process_request=web.process_request):
        await asyncio.Future()  # run forever


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    run()
