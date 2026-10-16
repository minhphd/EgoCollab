"""helper/collab/agents.py's step script on a stand-in env (no simulator).

    python -m pytest tests/test_agents.py -q
"""

from types import SimpleNamespace

from helper.collab.agents import Executor

TASK = {"steps": [{"obj": "milk", "rel": "inside", "target": "fridge", "closed": True},
                  {"obj": "cup", "rel": "ontop", "target": "table", "closed": False},
                  {"obj": "box", "rel": "ontop", "target": "table", "closed": False}],
        "orderings": []}
WORLD = SimpleNamespace(pos=lambda o: (0.0, 0.0, 0.9), support=lambda point, skip=(): "counter",  # all on the counter
                        held={"robot": None, "person": None})
ENV = SimpleNamespace(unwrapped=SimpleNamespace(task=TASK, robot_can=[True, True, False], w=WORLD))


def script(ex: Executor, v: int) -> list[str]:
    """The skills the executor runs for step v when every skill succeeds."""
    a, out, t = ex.start(0.0, v), [], 0.0
    while a["skill"] != "noop":
        out.append(a["skill"] + ("" if a["skill"] in ("pick", "place") else f" {a['object']}"))
        t += 1.0
        a = ex.follow(t, {"skill": "done"})
    return out


def test_a_container_with_a_door_is_opened_and_closed():
    assert script(Executor(ENV, 0), 0) == ["pick", "open fridge", "place", "close fridge"]
    assert script(Executor(ENV, 0), 1) == ["pick", "place"]


def test_a_failed_skill_blocks_the_step_for_15_s():
    ex = Executor(ENV, 0)
    ex.start(0.0, 1)
    a = ex.follow(3.0, {"skill": "failed: nothing in the gripper"})
    assert a["skill"] == "noop" and ex.cur is None and ex.blocked[1] == 18.0
    assert ex.segments == [[0.0, 3.0, 1]]


def test_steps_the_robot_cannot_do_are_blocked_for_good():
    assert ENV.unwrapped.robot_can[2] is False and Executor(ENV, 0).blocked[2] == float("inf")


def test_an_object_it_cannot_place_goes_back_where_it_came_from():
    ex = Executor(ENV, 0)
    ex.start(0.0, 1)
    a = ex.follow(1.0, {"skill": "done"})  # picked
    assert a["skill"] == "place" and a["target"] == "table"
    a = ex.follow(2.0, {"skill": "failed: no reachable free spot ontop table"})
    assert (a["skill"], a["rel"], a["target"]) == ("place", "ontop", "counter") and ex.blocked[1] == 17.0
    a = ex.follow(3.0, {"skill": "done"})
    assert a["skill"] == "noop" and ex.cur is None and ex.log[-1] == "3.0 returned 1"


def test_a_pick_that_fails_with_the_object_in_hand_goes_on_to_place_it():
    ex = Executor(ENV, 0)
    ex.start(0.0, 1)
    WORLD.held["robot"] = "cup"
    a = ex.follow(1.0, {"skill": "failed: {'arm': 0.0} not reached"})
    WORLD.held["robot"] = None
    assert (a["skill"], a["target"]) == ("place", "table") and ex.cur == 1 and 1 not in ex.blocked


def test_an_object_taken_before_the_robot_has_it_stops_the_trip():
    ex = Executor(ENV, 0)
    ex.start(0.0, 0)
    pos = WORLD.pos
    WORLD.pos = lambda o: (0.3, 0.0, 0.9)  # the person carries it off
    a = ex.follow(5.0, {"skill": "running"})
    WORLD.pos = pos
    assert a["skill"] == "stop" and ex.cur is None and ex.blocked[0] == 20.0 and ex.log[-1] == "5.0 taken 0"


def test_the_bip_agent_reads_clip_events(monkeypatch, tmp_path):
    """The env's clip events (sense.py) reach BIP: a clip read as place(milk, fridge) marks the milk subgoal done.
    Gemini is replaced by canned replies: a prior with no orderings, and no step lists (so the fallbacks)."""
    from helper.collab import bip
    from helper.collab.agents import Bip
    monkeypatch.setattr(bip, "gemini", lambda text, schema, P: ({"edges": []} if "edges" in schema["properties"] else None, 0.0))
    task = dict(TASK, title="agent task", nodes=["v0", "v1", "v2"], labels=["put the milk in the fridge", "put the cup on the table",
                                                                            "put the box on the table"],
                step_objects={"v0": ["milk", "fridge"], "v1": ["cup", "table"], "v2": ["box", "table"]})
    env = SimpleNamespace(unwrapped=SimpleNamespace(task=task, robot_can=[True, True, False], w=WORLD))
    agent = Bip(env, 0, P=bip.Params(seed=0, cache=str(tmp_path)))
    agent.reset({}, {})
    assert [s["verb"] for s in agent.bel.R[0]["steps"]] == ["pick", "open", "place", "close"]  # the fridge has a door
    place = [0.02, 0.92, 0.02, 0.02, 0.02]
    agent.observe(3.0, [{"t": 2.5, "kind": "fixation", "object": "milk", "visible": ["milk", "fridge", "cup"]},
                        {"t": 3.0, "kind": "clip", "verbs": place, "object": "milk", "receptacle": "fridge",
                         "visible": ["milk", "fridge", "cup", "table"]}])
    assert agent.bel.done()[0] > 0.8 and agent.bel.done()[1] < 0.2
