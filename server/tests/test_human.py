"""Human matchmaking over real sockets, with replay and rating isolation."""
import asyncio
import json
import time
from contextlib import suppress

import cr_sim
import websockets

from cr_server import match, server
from test_web import Bot, DECK, _entry, _get, _serve
from cr_server.matchlog import MatchLog


def test_human_game_is_realtime_saved_and_does_not_touch_rankings(tmp_path, monkeypatch):
    original = match.run_match

    async def short(sim, a, b, **kw):
        return await original(sim, a, b, max_ticks=24, **kw)

    monkeypatch.setattr(match, "run_match", short)

    async def body(url, arena):
        async with websockets.connect(url) as bot:
            await bot.send(json.dumps({"t": "join", "name": "waiting-bot", "deck": DECK}))
            assert json.loads(await bot.recv())["t"] == "joined"
            queue = arena.matchmaking_snapshot()["waiting"]
            assert len(queue) == 1 and queue[0]["name"] == "waiting-bot"
            before = arena.board.standings()
            bot_messages = []

            async def answer_bot():
                async for raw in bot:
                    msg = json.loads(raw)
                    bot_messages.append(msg)
                    if msg["t"] == "obs":
                        await bot.send(json.dumps({"t": "act", "tick": msg["obs"]["tick"], "action": None}))

            answering = asyncio.create_task(answer_bot())
            try:
                async with websockets.connect(url) as human:
                    await human.send(json.dumps({"t": "play", "bot": queue[0]["id"], "name": "tester", "deck": DECK}))
                    started = time.monotonic()
                    messages = []
                    async for raw in human:
                        msg = json.loads(raw)
                        messages.append(msg)
                        if msg["t"] == "start":
                            assert msg["rated"] is False and msg["realtime"] is True
                            assert msg["arena"]["tps"] == 20 and "seed" not in msg
                        if msg["t"] == "obs" and msg["obs"]["tick"] == 0:
                            obs = msg["obs"]
                            slot = next(i for i, ok in enumerate(obs["legal"]) if ok)
                            await human.send(json.dumps({"t": "act", "tick": obs["tick"],
                                                         "action": [slot, 3.5, 10.5], "card": obs["hand"][slot]}))
                            assert arena.matchmaking_snapshot()["waiting"] == []
                            # A second human cannot reserve the same bot.
                            async with websockets.connect(url) as other:
                                await other.send(json.dumps({"t": "play", "bot": queue[0]["id"]}))
                                assert json.loads(await other.recv())["t"] == "error"
                        if msg["t"] == "result":
                            assert time.monotonic() - started >= 24 / cr_sim.TICKS_PER_SECOND
                            assert msg["rated"] is False and msg["delta"] == 0 and msg["rating"] is None
                            break
                    assert [m["obs"]["tick"] for m in messages if m["t"] == "obs"] == list(range(24))
                await asyncio.sleep(0)
                assert arena.board.standings() == before
                assert len(arena.matchmaking_snapshot()["waiting"]) == 1
                entry = arena.log.matches[0]
                assert entry["rated"] is False and entry["delta"] == 0
                saved = json.loads((tmp_path / "replays" / entry["file"]).read_text())
                assert saved["rated"] is False and sum(saved["human"]) == 1
                human_seat = saved["human"].index(True)
                assert len(saved["actions"]) == 1 and saved["actions"][0][1] == human_seat
                assert match.trace(cr_sim, saved)["meta"]["faithful"] is True
                http_url = url.replace("ws://", "http://")
                for path in ("/play", "/arena.js", "/api/matchmaking", "/api/replay/1"):
                    code, ctype, data = await asyncio.to_thread(_get, http_url + path)
                    assert code == 200, path
                    if path == "/api/replay/1":
                        assert json.loads(data)["match"]["rated"] is False
                assert any(m["t"] == "result" and m["delta"] == 0 for m in bot_messages)
            finally:
                answering.cancel()
                with suppress(asyncio.CancelledError):
                    await answering

    _serve(tmp_path, monkeypatch, 0.2, body)


def test_human_disconnect_saves_partial_replay_and_requeues_bot(tmp_path, monkeypatch):
    async def body(url, arena):
        async with websockets.connect(url) as bot:
            await bot.send(json.dumps({"t": "join", "name": "bot", "deck": DECK}))
            await bot.recv()
            bot_id = arena.matchmaking_snapshot()["waiting"][0]["id"]

            async def answer_bot():
                async for raw in bot:
                    msg = json.loads(raw)
                    if msg["t"] == "obs":
                        await bot.send(json.dumps({"t": "act", "tick": msg["obs"]["tick"], "action": None}))
                    elif msg["t"] == "result":
                        return msg

            answering = asyncio.create_task(answer_bot())
            async with websockets.connect(url) as human:
                await human.send(json.dumps({"t": "play", "bot": bot_id}))
                while True:
                    msg = json.loads(await human.recv())
                    if msg["t"] == "obs" and msg["obs"]["tick"] >= 2:
                        break
            result = await answering
            assert result["result"] == "win" and result["delta"] == 0
            assert arena.board.get("bot").games == 0
            entry = arena.log.matches[0]
            assert entry["forfeit"] and entry["rated"] is False and entry["ticks"] < 10
            saved = json.loads((tmp_path / "replays" / entry["file"]).read_text())
            assert match.trace(cr_sim, saved)["meta"]["faithful"]
            assert [b["id"] for b in arena.matchmaking_snapshot()["waiting"]] == [bot_id]

    _serve(tmp_path, monkeypatch, 0.2, body)


def test_unrated_games_do_not_enter_ranked_summaries_even_after_restart(tmp_path):
    log = MatchLog(str(tmp_path))
    rated = _entry(1, "a", "b", 1, 16.0, 1216.0, 1184.0)
    log.add(rated)
    rows = [{"rank": 1, "name": "a", "rating": 1216.0, "wins": 1, "losses": 0, "draws": 0, "games": 1}]
    before = log.snapshot(rows, set(), 0)["rows"]
    log.add({**_entry(2, "a", "Human · a", 2, 0, 1216.0, None), "rated": False, "human": [False, True]})
    for current in (log, MatchLog(str(tmp_path))):
        snap = current.snapshot(rows, set(), 0)
        assert snap["rows"] == before
        assert snap["matches"][0]["rated"] is False and snap["matches"][0]["replay"]
        assert "Human · a" not in current.agents


def test_silent_bot_forfeits_but_human_game_is_saved(tmp_path, monkeypatch):
    async def body(url, arena):
        async with websockets.connect(url) as bot:
            await bot.send(json.dumps({"t": "join", "name": "silent", "deck": DECK}))
            await bot.recv()
            bot_id = arena.matchmaking_snapshot()["waiting"][0]["id"]
            async with websockets.connect(url) as human:
                # Illegal decks must not consume the selected bot.
                await human.send(json.dumps({"t": "play", "bot": bot_id, "deck": ["knight"] * 8}))
                assert json.loads(await human.recv())["t"] == "error"
            assert arena.matchmaking_snapshot()["waiting"][0]["id"] == bot_id
            async with websockets.connect(url) as human:
                await human.send(json.dumps({"t": "play", "bot": bot_id}))
                async for raw in human:
                    msg = json.loads(raw)
                    if msg["t"] == "result":
                        assert msg["result"] == "win" and msg["delta"] == 0
                        break
                else:
                    raise AssertionError("human did not receive a result")
            entry = arena.log.matches[0]
            assert entry["forfeit"] and entry["rated"] is False
            assert arena.board.get("silent").games == 0
            saved = json.loads((tmp_path / "replays" / entry["file"]).read_text())
            trace = match.trace(cr_sim, saved)
            assert trace["meta"]["faithful"] and trace["frames"][-1][0] == entry["ticks"]

    _serve(tmp_path, monkeypatch, 0.01, body)


def test_zero_tick_forfeit_replays_the_empty_game():
    record = {"seed": 3, "actions": [], "frame": "seat", "ticks": 0,
              "deck0": DECK, "deck1": DECK, "outcome": 1, "sim_outcome": 0, "crowns": [0, 0]}
    trace = match.trace(cr_sim, record)
    assert trace["meta"]["faithful"] and trace["meta"]["ticks"] == 0
    assert len(trace["frames"]) == 1


def test_realtime_ticks_do_not_catch_up_after_a_slow_turn():
    asked = []

    class SlowOnce(Bot):
        async def act(self, obs, timeout):
            asked.append(time.monotonic())
            if obs["tick"] == 1:
                await asyncio.sleep(0.15)
            return None, None

    asyncio.run(match.run_match(cr_sim, SlowOnce(1), Bot(2), seed=3, max_ticks=5, realtime=True))
    assert all(b - a >= 0.045 for a, b in zip(asked, asked[1:]))
    assert asked[2] - asked[1] >= 0.15
