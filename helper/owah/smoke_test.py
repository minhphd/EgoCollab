"""Smoke test for Online Watch-And-Help: the VirtualHome API, then one helping episode with MCTS agents.

On a compute node, start the simulator (helper/owah/start_sim.sh, port 8080), then:
    ~/.conda/envs/owah/bin/python helper/owah/smoke_test.py [episode] [max_steps]
"""

import os
import pickle
import sys
import time

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../third_party/owah")
sys.path += [f"{here}/virtualhome/simulation", f"{here}/online_watch_and_help"]
os.chdir(f"{here}/online_watch_and_help")

from unity_simulator.comm_unity import UnityCommunication  # noqa: E402

episode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

comm = UnityCommunication(port="8080")
t = time.time()
ok = comm.reset(0)
_, graph = comm.environment_graph()
print(f"[api] reset {ok}, scene 0 graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges ({time.time() - t:.1f} s)")

from agents import MCTS_agent  # noqa: E402
from algos.arena_mp2 import ArenaMP  # noqa: E402
from envs.unity_environment import UnityEnvironment  # noqa: E402

tasks = pickle.load(open("dataset/test_env_set_help.pik", "rb"))
for env in tasks:  # the repo's own scripts drop these objects and lift forks before running in Unity
    g = env["init_graph"]
    gone = {n["id"] for n in g["nodes"] if n["class_name"] in ("garbagecan", "clothespile")}
    g["nodes"] = [n for n in g["nodes"] if n["id"] not in gone]
    g["edges"] = [e for e in g["edges"] if e["from_id"] not in gone and e["to_id"] not in gone]
    for n in g["nodes"]:
        if n["class_name"] == "cutleryfork":
            n["obj_transform"]["position"][1] += 0.1
task = tasks[episode]
print(f"[owah] {len(tasks)} test episodes; episode {episode}: task {task['task_name']}, env {task['env_id']}, goal {task['task_goal'][0]}")

env_fn = lambda env_id: UnityEnvironment(num_agents=2, max_episode_length=max_steps, port_id=env_id, env_task_set=tasks,
                                         observation_types=["partial", "partial"], use_editor=True, executable_args={},
                                         base_port=8080)
params = {"obs_type": "partial", "open_cost": 0, "should_close": False, "walk_cost": 0.05, "belief": {"forget_rate": 0}}
common = dict(recursive=False, max_episode_length=5, num_simulation=100, max_rollout_steps=5, c_init=0.1, c_base=1000000,
              num_samples=1, num_processes=1, logging=True, logging_graphs=False, agent_params=params)
alice, bob = dict(agent_id=1, char_index=0, **common), dict(agent_id=2, char_index=1, **common)
bob["recursive"] = True  # Bob plans with Alice's true goal
arena = ArenaMP(max_steps, 0, env_fn, [lambda x, y: MCTS_agent(**alice), lambda x, y: MCTS_agent(**bob)])
t = time.time()
arena.reset(episode)
success, steps, info = arena.run()
print(f"[owah] {'success' if success else 'not finished'} after {steps} steps in {time.time() - t:.0f} s")
print("[owah] Alice:", info["action"][0][:8])
print("[owah] Bob:  ", info["action"][1][:8])
