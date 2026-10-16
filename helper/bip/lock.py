"""Freeze BIP's settings: every field of Config, MindConfig and Costs in one file the live loop loads.

    python -m helper.bip.lock [key=value ...]        # writes helper/bip/locked.json
    run = locked_live(task)                          # Live with the frozen settings
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

from .config import Config, parse
from .decide import Costs
from .mind import MindConfig

PATH = Path(__file__).with_name("locked.json")
LIVE = ["action_tier=scene", "completion=scene", "look_s=3", "prior_model=gemini-3.5-flash"]  # the experiments' tier


def load(path: Path = PATH) -> tuple[Config, MindConfig, Costs]:
    d = json.loads(path.read_text())
    mind = d["mind"] | {"s2_band": tuple(d["mind"]["s2_band"])}
    return Config(**d["config"]), MindConfig(**mind), Costs(**d["costs"])


def locked_live(task: dict, path: Path = PATH, slow_model: str = "gemini-3.5-flash"):
    from .live import Live
    cfg, mind, costs = load(path)
    return Live(cfg, task, mind, costs, slow_model)


def main() -> None:
    cfg = parse(LIVE + sys.argv[1:], Config)
    names = {f.name for f in fields(Config)}
    assert all(a.split("=")[0] in names for a in sys.argv[1:]), "only Config fields can be set here"
    PATH.write_text(json.dumps({"config": asdict(cfg), "mind": asdict(MindConfig()), "costs": asdict(Costs())}, indent=1))
    print(PATH)


if __name__ == "__main__":
    main()
