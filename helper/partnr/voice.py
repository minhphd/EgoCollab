"""The person's voice: a small local LM that answers the robot's questions from what the person knows (the
instruction, their steps, what is done, what they are doing). It answers briefly and only what is asked.

The LM runs as an HTTP server in the egocollab env (Qwen3.5 needs a newer transformers than the simulator's
env); the simulator workers call it with the standard library.
    python -m helper.partnr.voice <model dir> [port]          # server, on a GPU node
    answer(question, state, port) -> str                      # client, from a simulator worker
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8771


def plain(label: str) -> str:
    """A step in plain words, without the simulator's instance numbers ("the fork 6" -> "the fork")."""
    return re.sub(r"\s\d+\b", "", label)


def step(spec: dict, rooms: dict, near: dict | None = None) -> str:
    """A step as the person knows it: the object, and the furniture by its room and neighbours ("on the table in
    the kitchen, next to the fridge")."""
    kind = lambda x: re.sub(r"_\d+$", "", x).replace("_", " ")
    obj = kind(spec["objects"][0]) if "objects" in spec else kind(spec["object"])
    room = kind(rooms.get(spec["to"], "")) if rooms.get(spec["to"]) else ""
    ref = f" next to the {kind(spec['ref'])}" if spec.get("ref") else ""
    by = [kind(x) for x in (near or {}).get(spec["to"], [])][:1]
    return (f"put the {obj} {spec['rel']} the {kind(spec['to'])}" + (f" in the {room}" if room else "")
            + (f" (the one next to the {by[0]})" if by else "") + ref)


def counted(steps: list) -> list[str]:
    """Steps in plain words, repeats counted ("put the plate on the table (2 times)")."""
    n = {}
    for x in steps:
        n[plain(x)] = n.get(plain(x), 0) + 1
    return [x + (f" ({k} times)" if k > 1 else "") for x, k in n.items()]


def prompt(question: str, state: dict) -> str:
    steps = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(counted(state["steps"]))) or "none right now"
    done = "; ".join(counted(state["done"])) or "nothing yet"
    now = ("working on: " + plain(state["current"])) if state.get("current") else "between steps"
    if state.get("partial"):  # works the task out as they go
        know = (f"You asked the robot: \"{state['instruction']}\" You work the task out as you go. The steps you have "
                f"worked out so far:\n{steps}\nYou have not worked out the rest yet, so beyond these steps you do not "
                "know what else the task needs or how many of anything.")
        rule = ("Use only the steps above. If the answer is not in them, say you are not sure yet; never guess a "
                "number or a step.")
    else:
        know = f"Your task: {state['instruction']} The steps you have in mind:\n{steps}"
        rule = "Do not list every step unless asked about the whole task."
    return (f"You are a person doing a household task at home, with a robot helper in the room. {know}\n"
            f"Done so far: {done}. Right now you are {now}.\n"
            f"The robot asks you: \"{question}\"\n"
            "Answer as a busy person would: one or two short sentences, in your own words, only what the question "
            f"asks. Name objects and furniture the way the steps do. {rule}")


def answer(question: str, state: dict, port: int = PORT, timeout: float = 120.0) -> str:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/answer", data=json.dumps({"prompt": prompt(question, state)}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["text"]


def serve(path: str, port: int = PORT) -> None:
    import threading

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["prompt"]
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            with lock, torch.no_grad():
                out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.7, top_p=0.9)
            reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            body = json.dumps({"text": reply}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f"voice on port {port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else PORT)
