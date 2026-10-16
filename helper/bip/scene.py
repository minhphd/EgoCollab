"""The robot's estimate of the scene as an object graph, built only from what it observes.

Nodes are furniture (a known static map), objects (unknown until observed) and the agents, H (the person)
and R (the robot). Edges: on(o, f), in(o, f), next_to(o, o') and holding(agent, o). Updates come from
detections in the person's ego view and the robot's view (triples), and from actions, the person's as
recognized and the robot's own; nothing is read from the simulator's state.

    s = Scene(furniture)
    s.see([("cup_1", "on", "table_2"), ("cup_1", "next_to", "plate_3"), ("H", "holding", "spoon_4")], t)
    s.act("H", ("Pick", "cup_1"), t)
    s.holds({"object": "cup_1", "rel": "on", "to": "table_2", "ref": "plate_3"})
"""

from __future__ import annotations


class Scene:
    def __init__(self, furniture: list[str]):
        self.furniture = set(furniture)
        self.parent: dict[str, tuple[str, str]] = {}  # object -> (on | in, furniture)
        self.near: dict[str, set] = {}  # object -> objects next to it
        self.held: dict[str, str] = {}  # agent -> object
        self.seen: dict[str, float] = {}  # object -> last time observed
        self.opened: set[str] = set()

    def holder(self, o: str) -> str | None:
        return next((a for a, x in self.held.items() if x == o), None)

    def known(self, o: str) -> bool:
        return o in self.parent or self.holder(o) is not None

    def _lift(self, o: str) -> None:
        """o leaves its place: no parent, no neighbours, no holder."""
        self.parent.pop(o, None)
        for x in self.near.pop(o, set()):
            self.near.get(x, set()).discard(o)
        if (a := self.holder(o)) is not None:
            del self.held[a]

    def _rest(self, o: str, rel: str, f: str) -> None:
        if self.parent.get(o, (None, None))[1] != f or self.holder(o) is not None:
            self._lift(o)  # moved: old neighbours no longer hold
        self.parent[o] = (rel, f)

    def see(self, triples, t: float) -> None:
        for a, rel, b in triples:
            if rel == "holding":
                self._lift(b)
                self.held[a] = b
                self.seen[b] = t
            elif rel in ("on", "in"):
                self._rest(a, rel, b)
                self.seen[a] = t
            elif rel == "next_to":
                self.near.setdefault(a, set()).add(b)
                self.near.setdefault(b, set()).add(a)

    def act(self, agent: str, action: tuple, t: float) -> None:
        verb, args = action[0], action[1:]
        if verb == "Pick":
            self._lift(args[0])
            self.held[agent] = args[0]
            self.seen[args[0]] = t
        elif verb == "Place":
            o, rel, f, ref = args
            self._lift(o)
            self.parent[o] = (rel, f)
            if ref:
                self.see([(o, "next_to", ref)], t)
            self.seen[o] = t
        elif verb == "Open":
            self.opened.add(args[0])
        elif verb == "Close":
            self.opened.discard(args[0])

    def holds(self, spec: dict) -> bool:
        """The subgoal's predicate. An instance subgoal: its object rests at its furniture, unheld, next to its
        reference if any. A category subgoal (spec["objects"], the j-th of its kind, j = spec["rank"]): j candidates rest at
        its furniture. The furniture is spec["to"], or any of spec["tos"]; on and in are not told apart. Next to is
        symmetric: with no ref on the furniture yet, a category candidate needs no neighbour (the later one does)."""
        tos = spec.get("tos") or [spec["to"]]
        if "objects" in spec:  # with refs on the furniture, a candidate counts only next to one of them
            here = lambda o: [r for r in spec.get("refs") or [] if r != o and self.parent.get(r, (None, None))[1] in tos]
            near = lambda o: not here(o) or any(r in self.near.get(o, ()) for r in here(o))
            return sum(1 for o in spec["objects"] if self.parent.get(o, (None, None))[1] in tos and near(o)) >= spec.get("rank", 1)
        p = self.parent.get(spec["object"])
        return p is not None and p[1] in tos and (not spec.get("ref") or spec["ref"] in self.near.get(spec["object"], ()))

    def placed_ref(self, spec: dict) -> str | None:
        """For a category subgoal with refs: one already on its furniture to place next to, else None."""
        tos = spec.get("tos") or [spec["to"]]
        return next((r for r in spec.get("refs") or [] if self.parent.get(r, (None, None))[1] in tos), None)

    def candidates(self, spec: dict) -> list[str]:
        """The objects that can serve the subgoal: its object, or a category subgoal's candidates."""
        return spec.get("objects") or [spec["object"]]
