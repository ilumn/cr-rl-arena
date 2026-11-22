# cr-arena-server

The tournament server for [`cr_sim`](../gym). Agents connect over WebSocket and
wait in a pool; the organiser decides what they play. Every game is
**server-authoritative**: the server owns the only real `cr_sim.Battle`, and
agents just answer observations with actions. Results update a persistent **Elo**
leaderboard, and every match's `(seed, action-log)` is saved for replay/audit.

Nothing starts by itself. Agents enrol by connecting, and then one of three
things gives them a game:

- **The tournament** (`/tournament`): the organiser draws a single-elimination
  bracket from the pool and starts each pairing's games by hand. See
  [Running a tournament](#running-a-tournament).
- **The open ladder**: a switch on the tournament page that pairs waiting agents
  continuously for rated practice games. Off by default.
- **A person** who picks a waiting bot on `/play`.

**Writing an agent?** See [CONNECTING.md](CONNECTING.md): how to connect, the
messages, the observation and action formats, turns and timing, and how playing on
the server differs from running the simulator locally.

## Run

```bash
pip install ./gym ./server      # server depends on the gym package (cr-sim)
cr-arena                        # or: python -m cr_server.server
```

Environment variables:

| var | default | meaning |
|---|---|---|
| `CR_ARENA_BIND` | `0.0.0.0:8765` | host:port to listen on |
| `CR_ARENA_DATA` | `./arena_data` | leaderboard, match log and replay storage |
| `CR_MOVE_TIMEOUT` | `2.0` | seconds an agent has to answer each tick (miss ⇒ no move that tick) |
| `CR_ADMIN_KEY` | random, printed at startup | the organiser's key for `/tournament`. Set it: a random one changes on every restart |
| `CR_AUTO_MATCH` | `0` | `1` starts a fresh server with the open ladder on. Once the organiser has used the switch, the saved choice wins |

## Protocol (JSON text frames over WebSocket)

```
C→S  {"t":"join","name":"myagent","deck":["knight",...8...]}
     {"t":"leaderboard"}                         # query standings, then close
S→C  {"t":"joined","name","rating"}
     {"t":"start","opponent"}
     {"t":"obs","obs":{...}}                      # sent every tick; reply required
C→S  {"t":"act","tick":N,"action":[slot,x,y] | null}   # N = obs.tick; slot 0-3
S→C  {"t":"result","result":"win"|"loss"|"draw","crowns":[mine,theirs],
      "rating","delta","match","seed"}
     {"t":"aborted","reason"}                     # match void: opponent stopped answering
     {"t":"error","msg"}                          # refused or removed; connection closes
```

- **Every agent plays as the bottom player.** `obs` and `action` are in the agent's
  own seat frame: its side is `y < 15`, its entities are owner `0`, crowns are
  `[mine, theirs]`. The simulator mirrors the board for whichever agent is really
  on top, so an agent is never told its seat and never needs it.
- The match **seed is sent only with the result.** Sent beforehand, it would let an
  agent reconstruct the opponent's hand order.

- **Every tick is a turn.** Each agent gets an observation on every tick and has
  `CR_MOVE_TIMEOUT` seconds to answer. No answer in time means no move for that
  agent on that tick; the other agent's move still counts.
- **Replies carry the tick they answer.** A late reply is recognised by its tick and
  dropped, so it is never taken for the next tick's move. A reply with no tick is
  refused.
- **Ten missed turns in a row** and the agent is disconnected. That match is closed
  and not recorded: no result, no rating change, no replay. The opponent gets
  `aborted` and is re-queued.
- A disconnected agent **forfeits** to a connected opponent; that match is recorded.
- **Ignored moves are reported.** `obs.last_action` says what became of the agent's
  previous move: `ok`, `none`, the simulator's reason it could not be carried out
  (`not_enough_elixir`, `not_your_side`, `bad_slot`, ...), or the server's
  (`malformed`, `missed`). The server counts each agent's ignored moves and the web
  page shows them.
- One bad agent can't stall or crash a match or the server.

The `obs` dict is `cr_sim.Battle.view(seat).observe()`: `elixir`, `hand`,
`hand_cost`, `next_card`, `legal` (per-slot affordability mask), `last_action`,
`crowns`, `entities` (full board), `tick`, `phase`. See [`examples/random_agent.py`](examples/random_agent.py)
for a complete client.

## Web page

The same port serves a web page, so opening the server's address in a browser
shows the standings. There is nothing extra to run or deploy.

| path | what it is |
|---|---|
| `/` | Live standings: top three, the full table, recent results. Refreshes every few seconds. Click an agent for its rating history, deck, ignored-move count and last matches. |
| `/tournament` | The tournament: the pool of enrolled agents, the bracket, and the game being played, live. The organiser's controls appear after entering the key. |
| `/play` | Human matchmaking: the waiting bot queue, a join button per bot, and a playable arena with an editable deck. Human games run at 20 ticks per real second, save replays, and never affect Elo or ranked records. |
| `/replay` | Replay viewer. With no `id` it opens the newest match; `?id=N` opens match `N`. Beside the arena, score and playback controls it lists recent matches (search by agent, filter by upsets, king-tower wins or close games) and can play the next one automatically. Add `&t=75` to open a match paused 75 seconds in. |
| `/api/standings` | The page's data as JSON (standings, recent results, live counts). |
| `/api/tournament` | The pool and the bracket as JSON. |
| `/api/matchmaking` | Available bot connection IDs, names, ratings and decks, plus playable cards and the default human deck. |
| `/api/replay/N` | Frame-by-frame trace of match `N`, re-simulated on request from the saved seed and actions. |
| `/healthz` | Liveness probe. |

Notes:

- **What is public.** Anyone who can reach the server can see every agent's name,
  rating, record and current deck, and can watch any saved match, including both
  players' elixir. Ladder and human games are not shown while they are being
  played. **Tournament games are**: anyone on `/tournament` sees the board and
  both decks as it happens, a fraction of a second behind. That is nothing an
  agent is not already told about its own game. Elixir is the exception: an
  agent never sees its opponent's, so during a game it is sent only to a screen
  signed in as the organiser.
- **Match log.** Besides `leaderboard.json` and the replay files, the server keeps
  `matches.jsonl` in `CR_ARENA_DATA`, one line per finished match. The page's
  results list, rating charts and rank movement come from it.
- **Human games.** Open `/play`, choose a name and deck, then click **Join game**
  beside a waiting bot. Select a hand card (or press 1–4), then click or tap the
  board to deploy. Your side is always the bottom. These games are marked
  `rated: false` in logs and replays, with zero rating delta; they do not enter
  ranked W/L/D, form, rating history, or move statistics. Closing the browser or
  leaving forfeits the game and saves its partial replay. The bot rejoins the
  normal queue afterward. Bots already in a game cannot be claimed.
- **Human timing.** The referee limits human games to the simulator's 20 ticks per
  second, counting bot response time toward each tick. Slower replies can slow a
  game, but never produce accelerated catch-up. Humans send clicks only; waiting
  without playing a card is allowed. Bot-versus-bot matches retain their normal
  fast tournament timing and tick-tagged protocol.
- **Replays and simulator versions.** A replay is re-run on the server's current
  simulator. If the result no longer matches what was recorded (the simulator
  changed since), the viewer says so instead of presenting a different game as the
  original.
- The pages load two fonts from Google Fonts; without internet access they fall
  back to system fonts.

## Running a tournament

1. **Set the key.** Give the server a `CR_ADMIN_KEY` (on Railway: the service's
   Variables). Without one it makes a random key at each start and prints it to
   the log.
2. **Collect the pool.** Contestants start their agents; each appears on
   `/tournament` as soon as it joins. Nothing is played. An agent that
   disconnects drops out of the pool and returns when it reconnects under the
   same name.
3. **Unlock the controls.** On `/tournament` press **Organiser** and enter the
   key. It is remembered in that browser until you sign out. **Sit out** removes
   an agent from the draw without disconnecting it.
4. **Draw the bracket.** Choose one game per pairing or a best of 3 or 5, and
   seeding by rating or at random, then **Draw the bracket**. Everyone connected
   and not sitting out is entered (2 to 64). When the field is not a power of
   two, the top seeds get byes.
5. **Run each pairing.** Every pairing with both players has a **Run game**
   button. It plays one game at the chosen speed: 1× is real time, as a person
   would watch it; 2× and 4× are faster; Fast is as quick as the agents answer.
   Several pairings can run at once. In a series, press it again for the next
   game. A drawn game counts for nobody, so the pairing needs another.
6. **If an agent is missing**, its pairing cannot be run. Wait for it to
   reconnect, or use **Walkover** to send the other player through.
7. **Present it.** Open `/tournament` on the projector and press **Full screen**.
   A game appears there the moment it starts, with the bracket underneath; with
   several running, click **Watch** on a pairing to switch. To show the elixir
   bars, sign in on the projector as the organiser and press **Hide controls**;
   screens that are not signed in get everything except elixir.
8. **Afterwards**, **New tournament** clears the bracket and reopens enrolment.
   The agents still connected are still in the pool.

Notes:

- Tournament games are ordinary rated matches: they move Elo, appear in the
  results on `/`, and have replays, linked from each pairing.
- The open ladder and `/play` leave bracket entrants alone while the bracket is
  being played, so nobody is busy elsewhere when their pairing is called.
- A game is **void**, and the pairing simply needs running again, if an agent
  misses ten turns in a row (it is disconnected) or both connections drop. An
  agent that disconnects mid-game **forfeits** that game to a connected opponent.
- The bracket is saved in `CR_ARENA_DATA/tournament.json` and survives a
  restart. A game that was being played when the server stopped is not recorded;
  run it again.
- The names are not protected: any connection can join under any name. Agree the
  names beforehand, and use **Sit out** for anything in the pool you do not
  recognise.

## Reproducing / auditing a match

Each saved replay in `CR_ARENA_DATA/replays/` (named by match id, e.g.
`00000042.json`) holds the `seed` and `actions`.
Re-simulate it deterministically:

```python
import cr_sim, json
from cr_server.match import replay
r = json.load(open("arena_data/replays/<id>.json"))
print(replay(cr_sim, r["seed"], r["actions"], r["deck0"], r["deck1"]))  # same outcome/crowns
```

Replay files record each action as the agent sent it, in that agent's seat frame.
In a replay file `outcome` and `crowns` are the referee's: `1` seat-0 win, `2`
seat-1 win, `3` draw, and `[seat 0, seat 1]`.


## Design notes

- **Why Python, not Rust, for the server:** it imports the same `cr_sim` native
  extension the students use, so there is exactly one implementation of the game
  logic and no cross-language drift. Determinism holds because the engine itself
  is fixed-point.
- **Scaling:** matches are `asyncio` tasks and run concurrently. A full 3-minute
  match is 3,600 round-trips per agent, so how long it takes in real time depends
  almost entirely on how quickly the agents answer.
