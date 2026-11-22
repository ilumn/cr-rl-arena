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
    CR_ADMIN_KEY         the organiser's key for the tournament page. If unset, a
                         random one is made at startup and printed to the log.
    CR_AUTO_MATCH        "1" pairs waiting agents by itself from the first start
                         (the open ladder). Default off: agents wait in the pool
                         until the organiser starts something. The organiser can
                         switch it from the tournament page; that choice is saved.

Protocol (JSON text frames)
  C->S  {"t":"join","name":str,"deck":[8 card names]}   (deck required; an
        illegal deck is refused with {"t":"error","msg":...} and the socket closed)
        {"t":"leaderboard"}                      (query then close)
  S->C  {"t":"joined","name","rating"}
        {"t":"start","opponent"}                  (plus "deck":[8 names] when the server
                                                  dealt this agent a deck for the game)
        {"t":"obs","obs":{...}}                   (every tick; reply required)
  C->S  {"t":"act","tick":obs.tick,"action":[slot,x,y]|null}
  S->C  {"t":"result","result":"win"|"loss"|"draw","crowns":[mine,theirs],
         "rating","delta","match","seed"}
        {"t":"aborted","reason"}                  (match void: the opponent stopped answering)
        {"t":"leaderboard","standings":[...]}

Joining puts an agent in the pool. What it is given to play is up to the
organiser: a pairing in the tournament bracket, the open ladder when that is
switched on, or a person who picks it from the play page. To the agent these are
all the same: a `start`, a run of `obs`, a `result`.

The tournament page talks to the server over the same socket (see `/tournament`):
  C->S  {"t":"watch"}                             first message: become a spectator
        {"t":"focus","pid":str|null}              which live game to be sent (null: the newest)
        {"t":"auth","key":str}                    become the organiser
        {"t":"admin","do":"draw"|"run"|"walkover"|"reset"|"auto"|"exclude", ...}
                                                  (draw: best_of, seeding, random_from)
  S->C  {"t":"state", ...}                        pool and bracket, on every change
        {"t":"live","pid","meta","p0","p1",...}   a game this spectator is now watching
        {"t":"frame","pid","f":[tick,crowns,elixir,entities],"ev":[deploys]}
                                                  (elixir is null unless this is the organiser)
        {"t":"over","pid", ...}                   that game's result
        {"t":"auth","ok"} / {"t":"admin","ok","msg"}

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
import hmac
import json
import os
import random
import secrets
import time
import uuid
from contextlib import suppress

import cr_sim
import websockets

from . import match
from .matchlog import UPSET_GAP, MatchLog
from .leaderboard import Leaderboard
from .tournament import MAX_ENTRANTS, Tournament
from .web import Web

BIND = os.environ.get("CR_ARENA_BIND", "0.0.0.0:8765")
DATA_DIR = os.environ.get("CR_ARENA_DATA", "./arena_data")
MOVE_TIMEOUT = float(os.environ.get("CR_MOVE_TIMEOUT", "2.0"))
ADMIN_KEY = os.environ.get("CR_ADMIN_KEY", "")
AUTO_MATCH = os.environ.get("CR_AUTO_MATCH", "0").strip().lower() in ("1", "true", "yes", "on")
SPEEDS = (1, 2, 4, 0)       # how fast a watched game runs; 0 is as fast as the agents answer
MAX_SPECTATORS = 400
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


class Spectator:
    """A browser on the tournament page: sent the bracket, and one live game."""

    def __init__(self, ws):
        self.ws = ws
        self.admin = False          # has given the organiser's key
        self.fails = 0
        self.focus = None           # the pairing it asked to watch; None follows the newest game
        self.watching = None        # the pairing whose frames it is being sent
        self.open = True
        self.q: asyncio.Queue = asyncio.Queue()

    def push(self, obj, droppable=False):
        """Queue a message without ever waiting on this browser. A slow one loses
        frames; one that cannot even keep up with the rest is closed."""
        if not self.open:
            return
        backlog = self.q.qsize()
        if droppable and backlog > 40:
            return
        if backlog > 400:
            self.open = False
            asyncio.ensure_future(self.ws.close())
            return
        self.q.put_nowait(obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":")))

    async def pump(self):
        try:
            while True:
                await self.ws.send(await self.q.get())
        except websockets.ConnectionClosed:
            pass
        finally:
            self.open = False


class LiveGame:
    """A tournament game in progress, as the spectators see it."""

    def __init__(self, arena, pairing, speed):
        self.arena, self.pid, self.speed = arena, pairing["id"], speed
        self.number = len(pairing["games"]) + 1
        self.dealt = arena.tour.dealt(pairing)      # None: own decks; "random" or "mirror": dealt by the server
        self.header = None          # sent to a spectator before any frame
        self.last = None            # the newest frame (public, organiser's), for a spectator who arrives mid-game
        self.pending = []           # deploys since the last frame went out
        self.sent_at = 0.0
        self.seats = None

    def begin(self, a, b):
        """Seats are drawn: `a` is the bottom player on the board."""
        self.seats = (a, b)
        cards = self.arena._deck_checker.deployable_cards() + ["princess_tower", "king_tower"]
        self.idx = {n: i for i, n in enumerate(cards)}
        self.cards = cards

    def on_tick(self, b, t, deploys, out):
        self.pending += deploys
        first = self.header is None
        if not (first or out or t % 2 == 0):
            return
        now = time.monotonic()
        if not (first or out) and now - self.sent_at < 0.03:
            return                  # a game run flat out is thinned to what a screen can show
        frame = match.spectator_frame(b, self.idx, t)
        if first:
            a, c = self.seats
            self.header = json.dumps({
                "t": "live", "pid": self.pid, "meta": match.board_meta(cr_sim, self.cards),
                "p0": a.name, "p1": c.name, "deck0": a.deck, "deck1": c.deck,
                "speed": self.speed, "game": self.number, "dealt": self.dealt,
                "towers": [e for e in frame[3] if e[6] & 2],
            }, separators=(",", ":"))
            self.arena._announce(self)
        # Elixir is the one thing on the board an agent is never told about its
        # opponent, so only the organiser's screen is sent it while a game is live.
        self.last = tuple(
            json.dumps({"t": "frame", "pid": self.pid, "f": f, "ev": self.pending}, separators=(",", ":"))
            for f in (frame[:2] + [None] + frame[3:], frame))
        self.pending, self.sent_at = [], now
        for sp in self.arena.spectators:
            if sp.watching == self.pid:
                sp.push(self.last[sp.admin], droppable=not out)


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
        self.agents: dict[str, list[WsAgent]] = {}   # name -> its open connections: the pool
        self.tour = Tournament(os.path.join(DATA_DIR, "tournament.json"), auto=AUTO_MATCH)
        self.admin_key = ADMIN_KEY or secrets.token_urlsafe(12)
        self.spectators: set[Spectator] = set()
        self.games: dict[str, LiveGame] = {}    # pairing id -> its game in progress
        self.featured = None                    # the newest of those
        self.notice = None                      # the last thing the organiser should hear about
        self._dirty = False
        self._tasks: set[asyncio.Task] = set()

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
        if msg.get("t") == "watch":
            await self._spectate(ws)
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
        self.agents.setdefault(name, []).append(agent)
        self._changed()
        try:
            await ws.wait_closed()
        finally:
            agent.alive = False
            self.idle.pop(agent.queue_id, None)
            self.agents[name].remove(agent)
            if not self.agents[name]:
                del self.agents[name]
            self._changed()
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
                        for ag in self.idle.values() if ag.alive and not self.tour.in_play(ag.name)],
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
        bot = self.idle.get(bot_id) if isinstance(bot_id, str) else None
        if bot is not None and self.tour.in_play(bot.name):
            await ws.send(json.dumps({"t": "error", "msg": "That bot is in the tournament bracket. Please choose another."}))
            return
        bot = self.idle.pop(bot_id, None) if bot is not None else None
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
            if not (self.tour.auto and self.tour.phase == "lobby"):
                continue                # nothing starts by itself unless the open ladder is on
            waiting = [ag for ag in self.idle.values() if ag.alive]
            for i in range(0, len(waiting) - 1, 2):
                a, b = waiting[i:i + 2]
                self.idle.pop(a.queue_id, None)
                self.idle.pop(b.queue_id, None)
                asyncio.create_task(self._play(a, b))

    async def _play(self, a: WsAgent, b: WsAgent, *, live: LiveGame | None = None, pace: float = 0.0):
        """Play one game and record it. Returns what came of it: `{"match": id,
        "winner": name or None, ...}` for a game that counted, `{"void": why}`
        for one that did not."""
        rated = not (a.human or b.human)
        # Seats are drawn at random for every match: `a` is player 0 (bottom) from
        # here on. Queue order used to decide it, so with two agents connected the
        # same one was player 0 in every game.
        if random.random() < 0.5:
            a, b = b, a
        self.live += 1
        self._changed()
        try:
            seed = random.getrandbits(63)
            for ag, opponent in ((a, b), (b, a)):
                start = {"t": "start", "opponent": opponent.name}
                if live is not None and live.dealt:
                    start["deck"] = ag.deck         # not the one it joined with: say so
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
            if live is not None:
                live.begin(a, b)
                options = {"pace": pace, "stop_if_gone": True, "watch": live.on_tick}
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
                    return {"void": f"{' and '.join(g.name for g in gone)} stopped responding"}
            out = res["outcome"]
            # forfeit: a disconnected agent loses to a connected one
            forfeit = False
            if not a.alive and b.alive:
                out, forfeit = 2, True
            elif not b.alive and a.alive:
                out, forfeit = 1, True
            elif not rated and not a.alive and not b.alive:
                out, forfeit = 3, True
            if live is not None and out == 0:
                return {"void": "both agents lost their connection"}

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
            return {"match": mid, "winner": a.name if out == 1 else b.name if out == 2 else None,
                    "p0": a.name, "p1": b.name, "crowns": c, "forfeit": forfeit}
        except Exception as e:  # never let one match kill the server
            print(f"[match error] {a.name} vs {b.name}: {e!r}")
            if not rated:
                for ag in (a, b):
                    await ag.send({"t": "error", "msg": "The game could not finish. Please try again."})
            return {"void": "the server could not finish the game"}
        finally:
            self.live -= 1
            for x in (a, b):
                self._enqueue(x)
            self._changed()

    # ---- the tournament: pool, bracket, spectators ---------------------------
    def tournament_snapshot(self) -> dict:
        """Everything the tournament page shows."""
        here = {n for n, ags in self.agents.items() if any(ag.alive for ag in ags)}
        free = {ag.name for ag in self.idle.values() if ag.alive}
        excluded = set(self.tour.d["excluded"])
        snap = self.tour.snapshot()
        for r in snap["rounds"]:
            r["pairings"] = [{**p, "online": [n in here for n in p["players"]]} for p in r["pairings"]]
        pool = [{"name": n, "rating": round(self.board.get(n).rating, 1), "deck": self.agents[n][-1].deck,
                 "busy": n not in free, "excluded": n in excluded} for n in here]
        pool.sort(key=lambda e: (-e["rating"], e["name"]))
        return {**snap, "pool": pool, "live": list(self.games), "featured": self.featured,
                "notice": self.notice, "max": MAX_ENTRANTS, "now": int(time.time() * 1000)}

    def _changed(self):
        """Tell the spectators, once, after whatever is changing has finished changing."""
        if not self._dirty:
            self._dirty = True
            asyncio.get_running_loop().call_soon(self._flush)

    def _flush(self):
        self._dirty = False
        if self.spectators:
            text = json.dumps({"t": "state", **self.tournament_snapshot()}, separators=(",", ":"))
            for sp in self.spectators:
                sp.push(text)

    def _say(self, text):
        self.notice = {"at": int(time.time() * 1000), "text": text}

    def _attach(self, sp: Spectator, pid):
        """Start sending `sp` the game in pairing `pid`, from where it is now."""
        game = self.games.get(pid)
        sp.watching = pid if game else None
        if game and game.header:
            sp.push(game.header)
            sp.push(game.last[sp.admin])

    def _announce(self, game: LiveGame):
        """A game's first frame is ready: show it to everyone not watching another."""
        for sp in self.spectators:
            if sp.focus == game.pid or sp.watching == game.pid or (sp.focus is None and sp.watching not in self.games):
                sp.watching = game.pid
                sp.push(game.header)

    async def _spectate(self, ws):
        if len(self.spectators) >= MAX_SPECTATORS:
            await ws.send(json.dumps({"t": "error", "msg": "Too many people are watching. Please try again shortly."}))
            return
        sp = Spectator(ws)
        self.spectators.add(sp)
        pump = asyncio.create_task(sp.pump())
        try:
            sp.push({"t": "state", **self.tournament_snapshot()})
            self._attach(sp, self.featured)
            async for raw in ws:
                try:
                    msg = json.loads(raw) if len(raw) <= 4096 else None
                except ValueError:
                    msg = None
                if not isinstance(msg, dict):
                    continue
                t = msg.get("t")
                if t == "focus":
                    pid = msg.get("pid")
                    sp.focus = pid if isinstance(pid, str) else None
                    self._attach(sp, sp.focus or self.featured)
                elif t == "auth":
                    key = msg.get("key")
                    sp.admin = isinstance(key, str) and hmac.compare_digest(key.encode(), self.admin_key.encode())
                    if not sp.admin:
                        sp.fails += 1
                        await asyncio.sleep(1)          # guessing is slow, and soon over
                    sp.push({"t": "auth", "ok": sp.admin})
                    if sp.fails >= 5:
                        break
                elif t == "admin":
                    problem = self._admin(msg) if sp.admin else "sign in as the organiser first"
                    sp.push({"t": "admin", "do": msg.get("do"), "ok": problem is None, "msg": problem})
        except websockets.ConnectionClosed:
            pass
        finally:
            self.spectators.discard(sp)
            sp.open = False
            pump.cancel()
            with suppress(asyncio.CancelledError):
                await pump

    def _admin(self, msg) -> str | None:
        """Carry out an organiser's command. Returns why it could not be done, or None."""
        do = msg.get("do")
        try:
            if do == "auto":
                self.tour.set_auto(msg.get("on") is True)
                self._queue_changed.set()
            elif do == "exclude":
                if not isinstance(msg.get("name"), str):
                    return "no such agent"
                self.tour.exclude(msg["name"], msg.get("on") is True)
            elif do == "draw":
                return self._draw(msg.get("best_of"), msg.get("seeding"), msg.get("random_from"))
            elif do == "run":
                return self._run_pairing(msg.get("pid"), msg.get("speed"))
            elif do == "walkover":
                self.tour.walkover(msg.get("pid"), msg.get("side"))
            elif do == "reset":
                self.tour.reset()
                self._queue_changed.set()
            else:
                return "unknown command"
        except ValueError as e:
            return str(e)
        finally:
            self._changed()
        return None

    def _random_deck(self) -> list[str]:
        """Eight cards at random that make a legal deck."""
        cards = self._deck_checker.deployable_cards()
        while True:
            deck = random.sample(cards, 8)
            if self._deck_problem(deck) is None:
                return deck

    def _draw(self, best_of, seeding, random_from=None) -> str | None:
        excluded = set(self.tour.d["excluded"])
        names = [n for n, ags in self.agents.items() if n not in excluded and any(ag.alive for ag in ags)]
        if seeding == "random":
            random.shuffle(names)
        else:
            names.sort(key=lambda n: (-self.board.get(n).rating, n))
        self.tour.draw(names, best_of, seeding, random_from)
        self.notice = None
        print(f"[tournament] bracket drawn: {len(names)} entrants, best of {best_of}, {seeding} seeding"
              + (f", random decks from game {random_from}" if random_from else ""))
        return None

    def _run_pairing(self, pid, speed) -> str | None:
        p = self.tour.pairing(pid)
        problem = self.tour.playable(p)
        if problem:
            return problem
        found = []
        for name in p["players"]:
            ag = next((x for x in self.idle.values() if x.alive and x.name == name), None)
            if ag is None:
                return f"{name} is in another game" if any(x.alive for x in self.agents.get(name, ())) \
                    else f"{name} is not connected"
            found.append(ag)
        # claimed with no await in between, so nothing else can take either agent
        for ag in found:
            self.idle.pop(ag.queue_id, None)
        p["live"] = True
        game = LiveGame(self, p, speed if speed in SPEEDS else 1)
        self.games[game.pid] = game
        self.featured = game.pid
        task = asyncio.create_task(self._tournament_game(p, found, game))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return None

    async def _tournament_game(self, p, agents, game: LiveGame):
        pid = p["id"]
        pace = 1 / (cr_sim.TICKS_PER_SECOND * game.speed) if game.speed else 0.0
        own = [ag.deck for ag in agents]
        if game.dealt:                          # for this game only
            mirror = self._random_deck()
            for ag in agents:
                ag.deck = list(mirror) if game.dealt == "mirror" else self._random_deck()
        try:
            out = await self._play(*agents, live=game, pace=pace)
        except BaseException:
            out = {"void": "the server could not finish the game"}
            raise
        finally:
            for ag, deck in zip(agents, own):
                ag.deck = deck
            self.games.pop(pid, None)
            if self.featured == pid:
                self.featured = next(reversed(self.games), None)
            if "void" in out:
                p["live"] = False
                over = {"t": "over", "pid": pid, "void": out["void"]}
                self._say(f"{p['players'][0]} vs {p['players'][1]}: game void, {out['void']}. Nothing was recorded.")
            else:
                crowns = out["crowns"] if out["p0"] == p["players"][0] else out["crowns"][::-1]
                self.tour.record(pid, out["winner"], {"match": out["match"], "crowns": crowns,
                                                      "forfeit": out["forfeit"], "dealt": game.dealt})
                over = {"t": "over", "pid": pid, "match": out["match"], "winner": out["winner"],
                        "crowns": out["crowns"], "forfeit": out["forfeit"], "wins": p["wins"],
                        "decided": p["winner"] is not None,
                        "champion": self.tour.d["champion"] if self.tour.phase == "done" else None}
            for sp in self.spectators:
                if sp.watching == pid:
                    sp.push(over)
                    sp.watching = None
            self._changed()

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
    if ADMIN_KEY:
        print("organiser key: taken from CR_ADMIN_KEY", flush=True)
    else:
        print(f"organiser key for /tournament (set CR_ADMIN_KEY to choose your own): {arena.admin_key}", flush=True)
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
