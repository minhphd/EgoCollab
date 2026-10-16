"""Local web viewer for egocentric episodes: video, gaze, task graph and current subgoal.

    python -m helper.viewer.server [--port 8765] [--episodes data/episodes]
"""

import argparse
import json
import mimetypes
import os
import re
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"


def list_episodes(root: Path) -> list[dict]:
    out = []
    for meta in sorted(root.glob("*/*/episode.json")):
        m = json.loads(meta.read_text())
        out.append({"path": f"{m['dataset']}/{m['id']}", "dataset": m["dataset"], "id": m["id"], "title": m["title"],
                    "video_id": m.get("video_id") or m["id"], "task": m.get("task", ""),
                    "start": (m.get("clip") or [0])[0]})
    return sorted(out, key=lambda e: (e["dataset"], e["video_id"], e["start"]))


class Handler(SimpleHTTPRequestHandler):
    """Serves static files, /api/episodes, and /data/<episode files> with HTTP range support."""

    def __init__(self, *args, episodes: Path, **kwargs):
        self.episodes = episodes
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        if self.path == "/api/episodes":
            return self._send(json.dumps(list_episodes(self.episodes)).encode(), "application/json")
        if self.path.startswith("/data/"):
            root = self.episodes.absolute()
            path = Path(os.path.normpath(root / unquote(self.path[len("/data/"):].split("?")[0])))
            if root not in path.parents or not path.is_file():  # video.mp4 symlinks may point outside root
                return self.send_error(HTTPStatus.NOT_FOUND)
            return self._send_file(path)
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")  # browsers revalidate, so code changes show on reload
        super().end_headers()

    def _send(self, body: bytes, ctype: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m:
            start = int(m[1]) if m[1] else size - int(m[2])
            end = int(m[2]) if m[1] and m[2] else size - 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if m else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining and (chunk := f.read(min(1 << 20, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser cancels range requests while seeking

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--episodes", type=Path, default=REPO / "data" / "episodes")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, episodes=args.episodes))
    print(f"Viewer on http://{args.host}:{args.port}  ({len(list_episodes(args.episodes))} episodes)")
    server.serve_forever()


if __name__ == "__main__":
    main()
