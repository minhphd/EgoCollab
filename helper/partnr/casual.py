"""What the person says at the start, before working out the details: each PARTNR instruction rewritten by Gemini as
one casual request with the overall goal and where, without the full list of objects, counts and furniture.

    python -m helper.partnr.casual <out.json> <episode ids file> [split=val]
The output maps episode id -> request, for Config.say_file.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

PROMPT = ("A person at home is about to do this household task and asks a helper robot for help. Write what they "
          "would say at the start, before they have worked out the details: one short, casual sentence with the "
          "overall goal and where, without listing every object, count or piece of furniture.\nTask: {task}\n"
          "Reply with the sentence only.")


def main(out: str, ids_file: str, split: str = "val") -> None:
    from helper.llm.gemini import Gemini, generate
    ids = [x for line in Path(ids_file).read_text().splitlines() if not line.startswith("#") for x in line.split()]
    eps = json.load(gzip.open(f"data/partnr/datasets/partnr_episodes/v0_0/{split}.json.gz"))["episodes"]
    title = {e["episode_id"]: e["instruction"] for e in eps if e["episode_id"] in ids}
    res = json.loads(Path(out).read_text()) if Path(out).exists() else {}
    for i in ids:
        if i not in res:
            reply, _, _ = generate(PROMPT.format(task=title[i]), cfg=Gemini(model="gemini-3.5-flash", thinking=""))
            res[i] = reply.strip().split("\n")[0].strip('"')
        print(f"{i}: {title[i][:90]}\n    -> {res[i]}")
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **kw)
