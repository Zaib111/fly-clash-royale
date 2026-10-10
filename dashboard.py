"""Training dashboard: live charts of a run's progress, read from runs/<name>/log.csv.

    python dashboard.py              # opens http://127.0.0.1:8766
    python dashboard.py --run fly1

Read-only and lightweight: it only reads the log files the trainer writes, so it is safe
to keep open while training runs.
"""
import argparse
import csv
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PAGE = ROOT / "web" / "dashboard.html"


def list_runs():
    if not RUNS.exists():
        return []
    runs = [d for d in RUNS.iterdir() if (d / "log.csv").exists()]
    return [d.name for d in sorted(runs, key=lambda d: (d / "log.csv").stat().st_mtime, reverse=True)]


def read_run(name):
    d = RUNS / name
    rows = []
    with open(d / "log.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("iter") == "iter":      # a repeated header line
                continue
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    args = {}
    if (d / "args.json").exists():
        args = json.loads((d / "args.json").read_text())
    out = RUNS / f"{name}.out"
    tail = out.read_text(errors="replace").splitlines()[-14:] if out.exists() else []
    return {"name": name, "rows": rows, "args": args, "tail": tail, "updated": (d / "log.csv").stat().st_mtime}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/runs":
            return self._send(200, json.dumps(list_runs()).encode(), "application/json")
        if u.path == "/api/run":
            name = parse_qs(u.query).get("name", [""])[0]
            if name not in list_runs():          # only names we listed; no path tricks
                return self._send(404, b"{}", "application/json")
            return self._send(200, json.dumps(read_run(name)).encode(), "application/json")
        self._send(404, b"not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run to show first (default: most recent)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/" + (f"?run={args.run}" if args.run else "")
    print(f"Training dashboard: {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
