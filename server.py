#!/usr/bin/env python3

import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "scores.db")
MODES = {"manual", "assist"}


def connect():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS best(pid TEXT, mode TEXT, nick TEXT, score INTEGER, lvl INTEGER, PRIMARY KEY(pid, mode))")
    return c


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/top":
            mode = parse_qs(url.query).get("mode", [""])[0]
            if mode not in MODES:
                return self.reply(400, {"error": "bad mode"})
            with connect() as c:
                rows = c.execute("SELECT nick, score, lvl FROM best WHERE mode=? ORDER BY score DESC LIMIT 50", (mode,)).fetchall()
            return self.reply(200, [{"nick": r[0], "score": r[1], "lvl": r[2]} for r in rows])
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "wumpus-world.html"), "rb") as f:
                return self.reply(200, f.read(), "text/html; charset=utf-8")
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/score":
            return self.reply(404, {"error": "not found"})
        try:
            size = int(self.headers.get("Content-Length", 0))
            if size > 2000:
                raise ValueError
            d = json.loads(self.rfile.read(size))
            pid, mode = str(d["pid"])[:40], d["mode"]
            score, lvl = int(d["score"]), int(d["lvl"])
            nick = (str(d["nick"]).strip()[:14]) or "Explorer"
            if not pid or mode not in MODES or not 0 < score <= 100000 or not 0 <= lvl <= 5:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            return self.reply(400, {"error": "bad request"})
        with connect() as c:
            c.execute(
                "INSERT INTO best VALUES(?,?,?,?,?) ON CONFLICT(pid, mode) DO UPDATE "
                "SET nick=excluded.nick, score=excluded.score, lvl=excluded.lvl WHERE excluded.score > best.score",
                (pid, mode, nick, score, lvl))
        self.reply(200, {"ok": True})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Open http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
