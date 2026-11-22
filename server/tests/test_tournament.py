"""The tournament: a self-enrolled pool, a bracket drawn on command, games started by hand."""
import asyncio
import json

import cr_sim
import pytest
import websockets

from cr_server import match, server, web
from cr_server.tournament import Tournament, round_name, seed_order
from test_web import Bot, DECK, _get

KEY = "organiser-key"


# ---------------------------------------------------------------------------
# The bracket by itself
# ---------------------------------------------------------------------------

def test_seed_order_keeps_the_top_seeds_apart():
    assert seed_order(2) == [1, 2] and seed_order(4) == [1, 4, 2, 3]
    assert seed_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert sorted(seed_order(32)) == list(range(1, 33))
    assert [round_name(i) for i in range(4)] == ["Final", "Semifinals", "Quarterfinals", "Round of 16"]


def test_byes_go_to_the_top_seeds_and_winners_advance(tmp_path):
    t = Tournament(str(tmp_path / "t.json"))
    with pytest.raises(ValueError):
        t.draw(["only"], 1, "rating")
    with pytest.raises(ValueError):
        t.draw(["a", "a"], 1, "rating")
    t.draw(list("abcde"), 1, "rating")                     # five entrants in a field of eight
    r1, r2, final = t.d["rounds"]
    assert [p["players"] for p in r1] == [["a", None], ["d", "e"], ["b", None], ["c", None]]
    assert [p["how"] for p in r1] == ["bye", None, "bye", "bye"]
    assert r2[0]["players"] == ["a", None] and r2[1]["players"] == ["b", "c"]
    assert t.playable(r2[0]) and t.playable(r1[0]) and t.playable(r1[1]) is None
    assert t.in_play("a") and t.in_play("e")

    t.record("0-1", "e", {"match": 1, "crowns": [0, 1], "forfeit": False})
    assert r1[1]["winner"] == 1 and r2[0]["players"] == ["a", "e"] and not t.in_play("d")
    t.record("1-0", None, {"match": 2, "crowns": [1, 1], "forfeit": False})    # a draw decides nothing
    assert r2[0]["winner"] is None and r2[0]["wins"] == [0, 0] and len(r2[0]["games"]) == 1
    t.record("1-0", "a", {"match": 3, "crowns": [2, 0], "forfeit": False})
    t.walkover("1-1", 1)
    assert final[0]["players"] == ["a", "c"] and r2[1]["how"] == "walkover"
    with pytest.raises(ValueError):
        t.walkover("1-1", 0)                               # already decided
    t.record("2-0", "c", {"match": 4, "crowns": [0, 3], "forfeit": False})
    assert t.phase == "done" and t.d["champion"] == "c" and not t.in_play("c")

    again = Tournament(str(tmp_path / "t.json"))           # it survives a restart
    assert again.phase == "done" and again.d["champion"] == "c" and len(again.d["rounds"]) == 3
    again.reset()
    assert again.phase == "lobby" and again.d["rounds"] == []


def test_later_games_of_a_series_can_be_dealt_random_decks(tmp_path):
    t = Tournament(str(tmp_path / "t.json"))
    with pytest.raises(ValueError):
        t.draw(["a", "b"], 3, "rating", random_from=4)
    t.draw(["a", "b"], 5, "rating", random_from=3)
    p, seen = t.pairing("0-0"), []
    for winner in ("a", "b", None, "a", "a"):                # the drawn third game still counts as game three
        seen.append(t.dealt(p))
        t.record("0-0", winner, {"match": 1, "crowns": [1, 0], "forfeit": False})
    assert seen == [None, None, "random", "random", "mirror"] and t.phase == "done"
    assert Tournament(str(tmp_path / "t.json")).snapshot()["random_from"] == 3


def test_a_series_needs_a_majority_of_its_games(tmp_path):
    t = Tournament(str(tmp_path / "t.json"))
    t.draw(["a", "b"], 3, "random")
    for winner, decided in (("a", False), ("b", False), ("b", True)):
        t.record("0-0", winner, {"match": 1, "crowns": [1, 0], "forfeit": False})
        assert (t.phase == "done") is decided
    assert t.d["champion"] == "b" and t.d["rounds"][0][0]["wins"] == [1, 2]


# ---------------------------------------------------------------------------
# Watching a game as it is played
# ---------------------------------------------------------------------------

def test_the_live_feed_matches_the_replay_of_the_same_game():
    seen = []

    def watch(b, t, deploys, out):
        seen.append((t, deploys, out))

    res = asyncio.run(match.run_match(cr_sim, Bot(7), Bot(8), seed=7, watch=watch))
    assert [t for t, _, _ in seen] == list(range(1, res["ticks"] + 1)) and seen[-1][2] == res["outcome"]
    live = [d for _, ds, _ in seen for d in ds]
    tr = match.trace(cr_sim, {**res, "deck0": DECK, "deck1": DECK, "sim_outcome": res["outcome"]})
    assert live == tr["events"] and live                   # same deploys, same places, as the replay shows


def test_a_paced_game_takes_at_least_its_time():
    async def timed():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await match.run_match(cr_sim, Bot(1), Bot(2), seed=1, max_ticks=20, pace=0.01)
        return loop.time() - t0

    assert asyncio.run(timed()) >= 0.2


# ---------------------------------------------------------------------------
# The whole thing over real sockets
# ---------------------------------------------------------------------------

async def _agent(url, name, games, starts=None):
    """Join the pool and answer every observation until the socket is closed."""
    bot = Bot(sum(map(ord, name)), name)
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"t": "join", "name": name, "deck": DECK}))
        try:
            async for raw in ws:
                m = json.loads(raw)
                if m["t"] == "obs":
                    await ws.send(json.dumps({"t": "act", "tick": m["obs"]["tick"], "action": bot.choose(m["obs"])}))
                elif m["t"] in ("start", "result"):
                    games.append((name, m["t"]))
                    if starts is not None and m["t"] == "start":
                        starts.append(m)
        except websockets.ConnectionClosed:
            pass


class Page:
    """The tournament page's side of the socket."""

    def __init__(self, ws):
        self.ws, self.state, self.seen = ws, None, []

    async def until(self, want, timeout=30):
        """Read until a message for which `want(msg)` is true, keeping the newest state."""
        async def go():
            async for raw in self.ws:
                m = json.loads(raw)
                self.seen.append(m)
                if m["t"] == "state":
                    self.state = m
                if want(m):
                    return m
        return await asyncio.wait_for(go(), timeout)

    async def do(self, **cmd):
        await self.ws.send(json.dumps({"t": "admin", **cmd}))
        return await self.until(lambda m: m["t"] == "admin")

    def pairing(self, pid):
        return next(p for r in self.state["rounds"] for p in r["pairings"] if p["id"] == pid)


def test_pool_bracket_and_hand_started_games(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "ADMIN_KEY", KEY)
    games = []

    async def main():
        arena = server.Arena()
        mm = asyncio.create_task(arena.matchmaker())
        async with websockets.serve(arena.handler, "127.0.0.1", 0,
                                    process_request=web.Web(arena).process_request) as srv:
            port = srv.sockets[0].getsockname()[1]
            url = f"ws://127.0.0.1:{port}"
            names = ["ada", "bo", "cy"]
            bots = [asyncio.create_task(_agent(url, n, games)) for n in names]
            async with websockets.connect(url) as ws, websockets.connect(url) as ws2:
                page, guest = Page(ws), Page(ws2)
                await ws.send(json.dumps({"t": "watch"}))
                await ws2.send(json.dumps({"t": "watch"}))
                await page.until(lambda m: m["t"] == "state" and len(m["pool"]) == 3)
                assert page.state["phase"] == "lobby" and not page.state["auto"]

                # enrolling starts nothing
                await asyncio.sleep(0.3)
                assert games == [] and arena.live == 0 and arena.log.matches == []

                # only the organiser can do anything
                assert (await page.do(do="draw", best_of=1, seeding="rating"))["ok"] is False
                await ws.send(json.dumps({"t": "auth", "key": "wrong"}))
                assert (await page.until(lambda m: m["t"] == "auth"))["ok"] is False
                await ws.send(json.dumps({"t": "auth", "key": KEY}))
                assert (await page.until(lambda m: m["t"] == "auth"))["ok"] is True

                # sit one out, then bring it back
                assert (await page.do(do="exclude", name="cy", on=True))["ok"]
                await page.until(lambda m: m["t"] == "state" and any(e["excluded"] for e in m["pool"]))
                assert (await page.do(do="exclude", name="cy", on=False))["ok"]
                assert (await page.do(do="draw", best_of=7, seeding="rating"))["ok"] is False
                assert (await page.do(do="draw", best_of=1, seeding="rating"))["ok"]
                await page.until(lambda m: m["t"] == "state" and m["phase"] == "running")
                r1, final = page.state["rounds"]
                assert [r["name"] for r in page.state["rounds"]] == ["Semifinals", "Final"]
                assert r1["pairings"][0]["how"] == "bye" and None not in r1["pairings"][1]["players"]
                assert all(r1["pairings"][1]["online"])

                # a drawn bracket still plays nothing by itself
                await asyncio.sleep(0.3)
                assert games == [] and arena.live == 0
                st, _, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{port}/api/tournament")
                assert st == 200 and json.loads(body)["phase"] == "running"
                # ...and its entrants are not offered to people on the play page
                assert arena.matchmaking_snapshot()["waiting"] == []

                assert (await page.do(do="run", pid="1-0", speed=0))["ok"] is False     # the final has one player
                assert (await page.do(do="run", pid="0-1", speed=0))["ok"]
                assert (await page.do(do="run", pid="0-1", speed=0))["ok"] is False     # already being played
                live = await guest.until(lambda m: m["t"] == "live")
                assert live["pid"] == "0-1" and {live["p0"], live["p1"]} == set(r1["pairings"][1]["players"])
                assert len(live["towers"]) == 6 and live["meta"]["tps"] == cr_sim.TICKS_PER_SECOND
                over = await guest.until(lambda m: m["t"] == "over")
                await page.until(lambda m: m["t"] == "over")
                frames = [m for m in guest.seen if m["t"] == "frame"]
                assert all(m["pid"] == "0-1" for m in frames)
                # the public feed never carries elixir; the organiser's does
                assert all(m["f"][2] is None for m in frames)
                mine = [m for m in page.seen if m["t"] == "frame"]
                assert mine and all(len(m["f"][2]) == 2 for m in mine)
                assert over["decided"] and over["match"] == 1 and over["champion"] is None
                # the last frame is the last tick, and what was shown is what the replay will show
                assert frames[-1]["f"][0] == arena.log.matches[0]["ticks"]
                assert frames[-1]["f"][1] == over["crowns"] == arena.log.matches[0]["crowns"]

                await page.until(lambda m: m["t"] == "state" and m["rounds"][1]["pairings"][0]["players"][1])
                semi = page.pairing("0-1")
                assert semi["winner"] is not None and semi["games"][0]["match"] == 1 and not semi["live"]
                assert arena.log.matches[0]["rated"] and (tmp_path / "replays" / "00000001.json").exists()

                # the final, with a walkover refused once it is decided
                assert (await page.do(do="run", pid="1-0", speed=0))["ok"]
                done = await page.until(lambda m: m["t"] == "state" and m["phase"] == "done")
                assert done["champion"] in names and len(arena.log.matches) == 2
                assert (await page.do(do="walkover", pid="1-0", side=0))["ok"] is False
                assert (await page.do(do="reset"))["ok"]
                await page.until(lambda m: m["t"] == "state" and m["phase"] == "lobby")
                assert len(page.state["pool"]) == 3            # everyone is still enrolled for the next one

                # an entrant who has gone cannot be started; the organiser can wave the other through
                assert (await page.do(do="draw", best_of=3, seeding="random"))["ok"]
                await page.until(lambda m: m["t"] == "state" and m["phase"] == "running")
                pid = next(p["id"] for p in page.state["rounds"][0]["pairings"] if p["how"] is None)
                gone = page.pairing(pid)["players"][0]
                bots[names.index(gone)].cancel()
                await page.until(lambda m: m["t"] == "state" and len(m["pool"]) == 2)
                no = await page.do(do="run", pid=pid, speed=0)
                assert no["ok"] is False and "not connected" in no["msg"]
                assert (await page.do(do="walkover", pid=pid, side=1))["ok"]
                await page.until(lambda m: m["t"] == "state" and m["rounds"][0]["pairings"][int(pid[-1])]["how"] == "walkover")

                st, ct, body = await asyncio.to_thread(_get, f"http://127.0.0.1:{port}/tournament")
                assert st == 200 and ct.startswith("text/html") and "<title>Tournament" in body.decode()
            for b in bots:
                b.cancel()
            await asyncio.gather(*bots, return_exceptions=True)
        mm.cancel()

    asyncio.run(main())
    assert games.count(("ada", "start")) + games.count(("bo", "start")) + games.count(("cy", "start")) == 4
    saved = json.loads((tmp_path / "tournament.json").read_text())
    assert saved["phase"] == "running" and saved["best_of"] == 3


def test_the_open_ladder_is_a_switch_the_organiser_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "ADMIN_KEY", KEY)
    original = match.run_match

    async def short(sim, a, b, **kw):
        return await original(sim, a, b, max_ticks=60, **kw)

    monkeypatch.setattr(match, "run_match", short)
    games = []

    async def main():
        arena = server.Arena()
        mm = asyncio.create_task(arena.matchmaker())
        async with websockets.serve(arena.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}"
            bots = [asyncio.create_task(_agent(url, n, games)) for n in ("ada", "bo")]
            async with websockets.connect(url) as ws:
                page = Page(ws)
                await ws.send(json.dumps({"t": "watch"}))
                await page.until(lambda m: m["t"] == "state" and len(m["pool"]) == 2)
                await asyncio.sleep(0.2)
                assert games == []
                await ws.send(json.dumps({"t": "auth", "key": KEY}))
                assert (await page.do(do="auto", on=True))["ok"]
                for _ in range(200):
                    if len(arena.log.matches) >= 2:
                        break
                    await asyncio.sleep(0.05)
                assert len(arena.log.matches) >= 2             # the ladder is running
                assert (await page.do(do="auto", on=False))["ok"]
                await asyncio.sleep(0.5)
                n = len(arena.log.matches)
                await asyncio.sleep(0.5)
                assert len(arena.log.matches) == n             # and now it is not
            for b in bots:
                b.cancel()
            await asyncio.gather(*bots, return_exceptions=True)
        mm.cancel()

    asyncio.run(main())
    assert Tournament(str(tmp_path / "tournament.json")).auto is False


def test_random_decks_are_dealt_told_and_put_back(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "ADMIN_KEY", KEY)
    games, starts = [], []

    async def main():
        arena = server.Arena()
        async with websockets.serve(arena.handler, "127.0.0.1", 0) as srv:
            url = f"ws://127.0.0.1:{srv.sockets[0].getsockname()[1]}"
            bots = [asyncio.create_task(_agent(url, n, games, starts)) for n in ("ada", "bo")]
            async with websockets.connect(url) as ws:
                page = Page(ws)
                await ws.send(json.dumps({"t": "watch"}))
                await page.until(lambda m: m["t"] == "state" and len(m["pool"]) == 2)
                await ws.send(json.dumps({"t": "auth", "key": KEY}))
                assert (await page.do(do="draw", best_of=3, seeding="rating", random_from=2))["ok"]
                for n in (1, 2, 3):
                    await page.until(lambda m: m["t"] == "state" and m["phase"] != "lobby" and not m["live"])
                    if page.state["phase"] == "done":
                        break
                    assert (await page.do(do="run", pid="0-0", speed=0))["ok"]
                    live = await page.until(lambda m: m["t"] == "live")
                    assert live["dealt"] == [None, "random", "mirror"][n - 1]
                    await page.until(lambda m: m["t"] == "over")
                # everyone has its own deck back once the game is over
                await page.until(lambda m: m["t"] == "state" and not m["live"])
                assert all(e["deck"] == DECK for e in page.state["pool"])
            for b in bots:
                b.cancel()
            await asyncio.gather(*bots, return_exceptions=True)
        return arena

    arena = asyncio.run(main())
    first, later = starts[:2], starts[2:]
    assert all("deck" not in m for m in first) and later
    for m in later:
        assert len(set(m["deck"])) == 8 and m["deck"] != DECK
        cr_sim.Battle().check_deck(m["deck"])
    assert later[0]["deck"] != later[1]["deck"]            # game two: a different random deck each
    if len(later) == 4:
        assert later[2]["deck"] == later[3]["deck"]        # the decider: one deck for both
    # what was dealt is what was played and saved, so the replay is the real game
    rec = json.loads((tmp_path / "replays" / "00000002.json").read_text())
    assert {tuple(rec["deck0"]), tuple(rec["deck1"])} == {tuple(m["deck"]) for m in later[:2]}
    assert match.trace(cr_sim, rec)["meta"]["faithful"]
    assert arena.tour.pairing("0-0")["games"][1]["dealt"] == "random"
