"""Object names per subgoal, as short nouns an open-vocabulary detector recognizes, from the task text alone.

    from helper.llm.vocab import objects
    objects("Dosa making", {"pan_hot": "Pan hot and oiled", "dosa_poured": "Dosa poured"})
    # {"pan_hot": ["frying pan", "oil bottle"], "dosa_poured": ["bowl", "frying pan", "ladle"]}
"""

import json

from .gemini import Gemini, generate


def objects(task: str, subgoals: dict[str, str], cfg: Gemini = Gemini(), max_names: int = 6) -> dict[str, list[str]]:
    """Subgoal id -> sorted lowercase object names; {} if the reply cannot be read."""
    ids = {k: f"S{i + 1}" for i, k in enumerate(subgoals)}
    lines = [f"Task: {task}", "Subgoals:"] + [f"{ids[k]}: {label}" for k, label in subgoals.items()]
    lines += ["", f"For every subgoal, list up to {max_names} physical objects a person handles or looks at while doing it, "
              "as short common nouns an object detector recognizes (for example \"frying pan\", \"spatula\", \"plate\"). "
              "Use the same name for the same object across subgoals."]
    sid = {"type": "string", "enum": list(ids.values())}
    schema = {"type": "object", "required": ["objects"], "properties": {"objects": {"type": "array", "items": {
        "type": "object", "required": ["subgoal", "names"],
        "properties": {"subgoal": sid, "names": {"type": "array", "maxItems": max_names, "items": {"type": "string"}}}}}}}
    text, _, _ = generate("\n".join(lines), schema=schema, cfg=cfg)
    back = {s: k for k, s in ids.items()}
    try:
        return {back[o["subgoal"]]: sorted({n.strip().lower() for n in o["names"] if n.strip()})
                for o in json.loads(text)["objects"]}
    except (ValueError, KeyError, TypeError):
        return {}
