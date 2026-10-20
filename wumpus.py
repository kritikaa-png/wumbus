#!/usr/bin/env python3
"""Wumpus World: engine, knowledge-based AI agent, CLI and tests (standard library only).

  python wumpus.py ai --level 2 --seed 7      watch the AI play (add --no-risk for safe-only)
  python wumpus.py play --level 1             play by hand: w/a/s/d move, f+dir shoot, g grab, c climb, q quit
  python wumpus.py test                       run the unit tests
  python wumpus.py scores                     show the local leaderboard (wumpus_scores.json)

Layout: Game = hidden world + rules + scoring. Agent = knowledge base + inference + BFS.
The Agent never holds a reference to the Game; it only receives percept dictionaries.
"""
import argparse, json, os, random, sys, unittest
from collections import deque

LEVELS = {1: ("Beginner", 4, 1, 2, 2, 1.0), 2: ("Explorer", 6, 1, 5, 2, 1.2),
          3: ("Adventurer", 8, 2, 9, 2, 1.5), 4: ("Expert", 10, 2, 15, 2, 2.0),
          5: ("Master", 12, 3, 24, 3, 3.0)}
PTS = dict(gold=1000, exit=1000, kill=200, cell=20, rev=5, bump=5, miss=50, die=500)
SCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wumpus_scores.json")

def neighbors(n, c):
    x, y = c
    return [(a, b) for a, b in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)) if 0 <= a < n and 0 <= b < n]

class Game:

    def __init__(self, size, wumpus, pits, arrows, mult=1.0, seed=None):
        self.n, self.mult = size, mult
        self.seed = random.randrange(10 ** 6) if seed is None else seed
        self._generate(wumpus, pits)
        self.pos, self.arrows, self.has_gold = (0, 0), arrows, False
        self.over = self.won = False
        self.visited = {(0, 0)}
        self.ledger = dict(Treasure=0, Exploration=0, Kills=0, Exit=0, Penalties=0)
        self.moves = self.revisits = self.shots = 0
        self.msg = ""

    @classmethod
    def from_level(cls, level, seed=None):
        _, n, w, p, a, m = LEVELS[level]
        return cls(n, w, p, a, m, seed)

    def _generate(self, w, p):
        n = self.n
        if not 4 <= n <= 20 or w < 1 or p < 0 or w + p > n * n - 4:
            raise ValueError("invalid configuration")
        for t in range(300):
            r = random.Random(self.seed + t * 7919)
            cells = [(x, y) for x in range(n) for y in range(n) if (x, y) != (0, 0)]
            r.shuffle(cells)
            gold = cells.pop()
            pool = [c for c in cells if c not in ((0, 1), (1, 0))]
            wum, pit = set(pool[:w]), set(pool[w:w + p])
            if self._reachable(gold, wum | pit):
                self.gold, self.wum, self.pit = gold, wum, pit
                return
        raise RuntimeError("could not generate a solvable world")

    def _reachable(self, goal, blocked):
        seen, q = {(0, 0)}, deque([(0, 0)])
        while q:
            c = q.popleft()
            if c == goal:
                return True
            for x in neighbors(self.n, c):
                if x not in seen and x not in blocked:
                    seen.add(x)
                    q.append(x)
        return False

    def percept(self, bump=False, scream=False):
        near = neighbors(self.n, self.pos)
        return dict(breeze=any(c in self.pit for c in near), stench=any(c in self.wum for c in near),
                    glitter=self.pos == self.gold and not self.has_gold, bump=bump, scream=scream,
                    arrows=self.arrows, has_gold=self.has_gold)

    def _add(self, key, label):
        self.ledger[key] += PTS[label] if key in ("Treasure", "Exploration", "Kills", "Exit") else -PTS[label]

    def _end(self, won, msg):
        self.over, self.won, self.msg = True, won, msg
        return self.percept()

    @property
    def base_score(self):
        return sum(self.ledger.values())

    @property
    def score(self):
        b = self.base_score
        return round(b * self.mult) if b > 0 else b

    def step(self, act, arg=None):
        if self.over:
            raise RuntimeError("game over")
        self.msg = ""
        bump = scream = False
        if act == "move":
            self.moves += 1
            nx, ny = self.pos[0] + arg[0], self.pos[1] + arg[1]
            if not (0 <= nx < self.n and 0 <= ny < self.n):
                bump = True
                self._add("Penalties", "bump")
            else:
                self.pos = (nx, ny)
                if self.pos in self.pit or self.pos in self.wum:
                    self._add("Penalties", "die")
                    return self._end(False, "You fell into a pit." if self.pos in self.pit else "The Wumpus ate you.")
                if self.pos not in self.visited:
                    self.visited.add(self.pos)
                    self._add("Exploration", "cell")
                elif not self.has_gold:
                    self.revisits += 1
                    self._add("Penalties", "rev")
        elif act == "shoot":
            if self.arrows < 1:
                self.msg = "No arrows left."
            else:
                self.arrows -= 1
                self.shots += 1
                x, y = self.pos
                while True:
                    x, y = x + arg[0], y + arg[1]
                    if not (0 <= x < self.n and 0 <= y < self.n):
                        break
                    if (x, y) in self.wum:
                        self.wum.discard((x, y))
                        scream = True
                        break
                self._add("Kills" if scream else "Penalties", "kill" if scream else "miss")
        elif act == "grab":
            if self.pos == self.gold and not self.has_gold:
                self.has_gold = True
                self._add("Treasure", "gold")
        elif act == "climb":
            if self.pos != (0, 0):
                self.msg = "You can only climb out at the entrance."
            elif self.has_gold:
                self._add("Exit", "exit")
                return self._end(True, "You escaped with the gold!")
            else:
                return self._end(False, "You left empty-handed.")
        return self.percept(bump, scream)

class Agent:

    def __init__(self, size, wumpus, arrows, allow_risk=True):
        self.n, self.W, self.risk_ok = size, wumpus, allow_risk
        self.pos, self.gold, self.arrows, self.kills = (0, 0), False, arrows, 0
        self.noP, self.noW, self.vis = set(), set(), {}
        self.deductions = 0

    def N(self, c):
        return neighbors(self.n, c)

    def no_w(self, c):
        return c in self.noW or self.kills >= self.W

    def safe(self, c):
        return c in self.noP and self.no_w(c)

    def observe(self, act, arg, pc):
        if act == "move" and not pc["bump"]:
            self.pos = (self.pos[0] + arg[0], self.pos[1] + arg[1])
        self.gold, self.arrows = pc["has_gold"], pc["arrows"]
        if pc["scream"]:
            self.kills += 1
            for v in self.vis.values():
                v["stale"] = True
            if act == "shoot":
                self.noW.add((self.pos[0] + arg[0], self.pos[1] + arg[1]))
        self.vis[self.pos] = dict(breeze=pc["breeze"], stench=pc["stench"], glitter=pc["glitter"])
        self.noP.add(self.pos)
        self.noW.add(self.pos)
        for c in self.N(self.pos):
            if not pc["breeze"]:
                self.noP.add(c)
            if not pc["stench"]:
                self.noW.add(c)

    def confirmed(self, key, ruled_out):
        out = set()
        for c, p in self.vis.items():
            if p[key] and not p.get("stale"):
                cand = [x for x in self.N(c) if not ruled_out(x)]
                if len(cand) == 1:
                    out.add(cand[0])
        return out

    def risk(self, c):
        a = b = 0.0
        for v in self.N(c):
            p = self.vis.get(v)
            if not p:
                continue
            if p["breeze"] and c not in self.noP:
                a = max(a, 1 / len([x for x in self.N(v) if x not in self.noP]))
            if p["stench"] and not self.no_w(c):
                b = max(b, 1 / len([x for x in self.N(v) if not self.no_w(x)]))
        return 1 - (1 - a) * (1 - b)

    def bfs(self, start, goal, ok):
        if goal(start):
            return []
        prev, q = {start: None}, deque([start])
        while q:
            c = q.popleft()
            for x in self.N(c):
                if x in prev or not ok(x):
                    continue
                prev[x] = c
                if goal(x):
                    path = []
                    while x != start:
                        path.append(x)
                        x = prev[x]
                    return path[::-1]
                q.append(x)
        return None

    def decide(self):
        here, cur = self.pos, self.vis.get(self.pos, {})

        def go(path, why):
            return ("move", (path[0][0] - here[0], path[0][1] - here[1]), why)

        if cur.get("glitter") and not self.gold:
            return ("grab", None, "Glitter: the gold is here, picking it up.")
        if self.gold:
            if here == (0, 0):
                return ("climb", None, "Gold in hand at the entrance: climbing out.")
            p = self.bfs(here, lambda c: c == (0, 0), self.safe)
            if p:
                return go(p, "Treasure acquired: BFS route home through proven-safe cells.")
        p = self.bfs(here, lambda c: c not in self.vis and self.safe(c), self.safe)
        if p:
            return go(p, f"Nearest unvisited proven-safe cell is {p[-1]} (BFS).")
        cw = self.confirmed("stench", self.no_w)
        if self.arrows > 0:
            for t in cw:
                adj = [c for c in self.N(t) if self.safe(c)]
                if here in adj:
                    return ("shoot", (t[0] - here[0], t[1] - here[1]), f"Wumpus confirmed at {t}: shooting.")
                q = self.bfs(here, lambda c: c in adj, self.safe)
                if q:
                    return go(q, f"Moving to a firing position against the Wumpus at {t}.")
        if self.risk_ok:
            cp = self.confirmed("breeze", lambda c: c in self.noP)
            cand = sorted({c for v in self.vis for c in self.N(v)
                           if c not in self.vis and not self.safe(c) and c not in cp and c not in cw}, key=self.risk)
            for c in cand:
                ng, r = [v for v in self.N(c) if v in self.vis], round(self.risk(c) * 100)
                if here in ng:
                    return ("move", (c[0] - here[0], c[1] - here[1]), f"Calculated risk: entering {c} (~{r}% estimated).")
                q = self.bfs(here, lambda x: x in ng, self.safe)
                if q:
                    return go(q, f"No safe cell left. Heading toward {c} for a calculated risk (~{r}%).")
        if here == (0, 0):
            return ("climb", None, "No safe progress possible: leaving without the gold.")
        p = self.bfs(here, lambda c: c == (0, 0), self.safe)
        return go(p, "No safe progress: returning to the entrance.") if p else ("climb", None, "Nothing left to do.")

def run_agent(game, agent, verbose=False):
    pc = game.percept()
    agent.observe(None, None, pc)
    for _ in range(30 * game.n * game.n):
        if game.over:
            break
        act, arg, why = agent.decide()
        pc = game.step(act, arg)
        agent.observe(act, arg, pc)
        if verbose:
            print(f"{act:5} {str(arg or ''):8} {why}\n{draw(game, agent)}\n")
    return game

def draw(game, agent=None, reveal=False):
    reveal = reveal or game.over
    rows = []
    for y in range(game.n - 1, -1, -1):
        row = []
        for x in range(game.n):
            c = (x, y)
            if c == game.pos: ch = "@"
            elif reveal and c in game.pit: ch = "P"
            elif reveal and c in game.wum: ch = "W"
            elif reveal and c == game.gold and not game.has_gold: ch = "$"
            elif c in game.visited: ch = "."
            elif agent and agent.safe(c): ch = "s"
            else: ch = "#"
            row.append(ch)
        rows.append(" ".join(row))
    return "\n".join(rows)

def record(game, level, mode):
    try:
        data = json.load(open(SCORE_FILE))
    except (OSError, ValueError):
        data = []
    data.append(dict(score=game.score, level=level, mode=mode, won=game.won, moves=game.moves, seed=game.seed))
    json.dump(data[-300:], open(SCORE_FILE, "w"))

def summary(game):
    print(draw(game, reveal=True))
    print(f"\n{game.msg}  Score {game.score} (base {game.base_score} x{game.mult})  Moves {game.moves}  Ledger {game.ledger}")

def cmd_play(a):
    g = Game.from_level(a.level, a.seed)
    names = {"w": (0, 1), "s": (0, -1), "a": (-1, 0), "d": (1, 0)}
    print(f"Seed {g.seed}. w/a/s/d move, f+dir shoot (e.g. fd), g grab, c climb, q quit.")
    while not g.over:
        pc = g.percept()
        print("\n" + draw(g) + "\nPercepts:", ", ".join(k for k in ("breeze", "stench", "glitter") if pc[k]) or "none", f"| arrows {g.arrows}")
        s = input("> ").strip().lower()
        if s == "q": return
        if s in names: g.step("move", names[s])
        elif len(s) == 2 and s[0] == "f" and s[1] in names: g.step("shoot", names[s[1]])
        elif s == "g": g.step("grab")
        elif s == "c": g.step("climb")
        if g.msg and not g.over: print(g.msg)
    summary(g)
    record(g, a.level, "manual")

def cmd_ai(a):
    g = Game.from_level(a.level, a.seed)
    _, n, w, _, arrows, _ = LEVELS[a.level]
    run_agent(g, Agent(n, w, arrows, not a.no_risk), verbose=True)
    summary(g)
    record(g, a.level, "ai")

def cmd_scores(_):
    try:
        data = sorted(json.load(open(SCORE_FILE)), key=lambda r: -r["score"])[:10]
    except (OSError, ValueError):
        data = []
    print("LOCAL LEADERBOARD")
    for i, r in enumerate(data, 1):
        print(f"{i:2}. {r['score']:6}  {LEVELS[r['level']][0]:10} {r['mode']:7} moves {r['moves']}")

class Tests(unittest.TestCase):
    def test_start_safe_and_solvable(self):
        for lv in LEVELS:
            for seed in range(40):
                g = Game.from_level(lv, seed)
                self.assertNotIn((0, 0), g.pit | g.wum | {g.gold})
                self.assertFalse({(0, 1), (1, 0)} & (g.pit | g.wum))
                self.assertTrue(g._reachable(g.gold, g.pit | g.wum))

    def test_percepts_match_world(self):
        g = Game(4, 1, 1, 2, seed=1)
        g.wum, g.pit, g.gold = {(1, 1)}, {(3, 3)}, (3, 0)
        g.pos = (1, 0)
        self.assertTrue(g.percept()["stench"]); self.assertFalse(g.percept()["breeze"])
        g.pos = (3, 2)
        self.assertTrue(g.percept()["breeze"])
        g.pos = (3, 0)
        self.assertTrue(g.percept()["glitter"])

    def test_wall_bump_and_no_exploration_farming(self):
        g = Game(4, 1, 1, 2, seed=2)
        self.assertTrue(g.step("move", (-1, 0))["bump"]); self.assertEqual(g.pos, (0, 0))
        g.step("move", (1, 0)); g.step("move", (-1, 0)); g.step("move", (1, 0))
        self.assertEqual(g.ledger["Exploration"], PTS["cell"])
        self.assertEqual(g.revisits, 2)

    def test_arrows_and_scream(self):
        g = Game(4, 1, 0, 2, seed=3)
        g.wum, g.pit, g.gold = {(0, 2)}, set(), (3, 3)
        self.assertTrue(g.step("shoot", (0, 1))["scream"])
        self.assertEqual((g.arrows, len(g.wum)), (1, 0))
        self.assertFalse(g.step("shoot", (1, 0))["scream"])
        self.assertEqual(g.ledger["Penalties"], -PTS["miss"])

    def test_winning_score(self):
        g = Game(4, 1, 0, 2, seed=4)
        g.wum, g.pit, g.gold = {(3, 3)}, set(), (1, 0)
        g.step("move", (1, 0)); g.step("grab"); g.step("move", (-1, 0)); g.step("climb")
        self.assertTrue(g.won)
        self.assertEqual(g.score, PTS["cell"] + PTS["gold"] + PTS["exit"])

    def test_bad_config_rejected(self):
        for cfg in ((3, 1, 1, 1), (6, 0, 1, 1), (4, 10, 10, 1)):
            with self.assertRaises(ValueError):
                Game(*cfg)

    def test_agent_has_no_access_to_world(self):
        a = Agent(4, 1, 2)
        self.assertFalse(any(isinstance(v, Game) for v in vars(a).values()))

    def test_safe_only_agent_never_dies(self):
        for lv in (1, 2, 3):
            for seed in range(60):
                g = Game.from_level(lv, seed)
                _, n, w, _, arrows, _ = LEVELS[lv]
                run_agent(g, Agent(n, w, arrows, allow_risk=False))
                self.assertTrue(g.over)
                self.assertFalse(g.pos in g.pit or g.pos in g.wum, f"level {lv} seed {seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("play", cmd_play), ("ai", cmd_ai)):
        sp = sub.add_parser(name); sp.add_argument("--level", type=int, default=1, choices=LEVELS)
        sp.add_argument("--seed", type=int); sp.set_defaults(fn=fn)
        if name == "ai": sp.add_argument("--no-risk", action="store_true")
    sub.add_parser("scores").set_defaults(fn=cmd_scores)
    sub.add_parser("test")
    args = ap.parse_args()
    if args.cmd == "test":
        unittest.main(argv=[sys.argv[0], "-v"])
    else:
        args.fn(args)
