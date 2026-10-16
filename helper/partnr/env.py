"""PARTNR environment for one worker: the Spot robot (agent 0) and the humanoid person (agent 1), driven by
our own controllers through PARTNR's oracle skills (Navigate, Pick, Place, Open, Close).

    sim = Sim(cfg, episode_ids)          # run from any directory; GPU node only (EGL rendering)
    sim.reset(episode_id)
    obs = sim.step({1: vector})          # per-agent low-level action vectors from Sim.skill
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np

from helper.bip.config import ROOT, Config

PARTNR = ROOT / "third_party/partnr/partnr-planner"


def episodes(cfg: Config) -> list[dict]:
    """Eligible episodes: rearrangement propositions only, at least min_subgoals of them."""
    path = PARTNR / f"data/datasets/partnr_episodes/v0_0/{cfg.split}.json.gz"
    out = []
    for e in json.load(gzip.open(path))["episodes"]:
        props = e["evaluation_propositions"]
        temporal = [c for c in e["evaluation_constraints"] if c["type"] == "TemporalConstraint"]
        ordered = bool(temporal and temporal[0]["args"].get("dag_edges"))
        ours = bool((e.get("info") or {}).get("family"))  # our families use counted placements (tasks.category_subgoals)
        if (len(props) >= cfg.min_subgoals and all(p["function_name"] in cfg.kinds for p in props)
                and (ours or all(p["args"].get("number", 1) == 1 for p in props)) and (ordered or not cfg.ordered_only)
                and (not cfg.only or e["episode_id"] in cfg.only)):
            out.append({"id": e["episode_id"], "ordered": ordered, "n": len(props)})
    return out


KEEP = {"agent_0": {"third_rgb_sensor", "head_panoptic_sensor"}, "agent_1": {"head_rgb_sensor", "head_panoptic_sensor", "third_rgb_sensor"}}
KEEP_OBS = {"agent_0_third_rgb", "agent_0_head_panoptic", "agent_1_head_rgb", "agent_1_head_panoptic", "agent_1_third_rgb"}


def prune_sensors(c) -> None:
    """Render only the person's head RGB and panoptic cameras, a panoptic camera on the robot's head (its view
    for the scene graph) and the two third-person views."""
    import habitat
    visual = ("rgb", "depth", "panoptic", "humanoid_detector")
    with habitat.config.read_write(c):
        robot = c.habitat.simulator.agents.agent_0.sim_sensors  # Spot has no head panoptic camera: copy the person's
        robot["head_panoptic_sensor"] = c.habitat.simulator.agents.agent_1.sim_sensors.head_panoptic_sensor
        c.habitat.gym.obs_keys = list(c.habitat.gym.obs_keys) + ["agent_0_head_panoptic"]
        c.habitat.gym.obs_keys = [k for k in c.habitat.gym.obs_keys if k in KEEP_OBS or not any(w in k for w in visual)]
        for agent in c.habitat.simulator.agents:
            sensors = c.habitat.simulator.agents[agent].sim_sensors
            c.habitat.simulator.agents[agent].sim_sensors = {k: v for k, v in sensors.items() if k in KEEP.get(agent, set())}
        if "humanoid_detector_sensor" in c.habitat.task.lab_sensors:
            del c.habitat.task.lab_sensors["humanoid_detector_sensor"]


class Sim:
    def __init__(self, cfg: Config, ids: list[str], out_dir: Path):
        from hydra import compose, initialize_config_dir
        from hydra.core.hydra_config import HydraConfig
        from omegaconf import open_dict

        from habitat_llm.agent.env import EnvironmentInterface, register_actions, register_measures, register_sensors
        from habitat_llm.agent.env.dataset import CollaborationDatasetV0
        from habitat_llm.utils import fix_config, setup_config
        from habitat_llm.utils.sim import init_agents

        os.chdir(PARTNR)  # the configs use paths relative to the repo
        px, a1 = cfg.head_px, "habitat.simulator.agents.agent_1.sim_sensors"
        overrides = [f"habitat.dataset.data_path=data/datasets/partnr_episodes/v0_0/{cfg.split}.json.gz",
                     "habitat.environment.iterator_options.shuffle=False",
                     f"habitat.environment.max_episode_steps={cfg.max_steps + 10}",
                     f"{a1}.head_rgb_sensor.height={px}", f"{a1}.head_rgb_sensor.width={px}",
                     f"paths.results_dir={out_dir}", f"evaluation.output_dir={out_dir}"]
        with initialize_config_dir(version_base=None, config_dir=str(PARTNR / "habitat_llm/conf")):
            c = compose(config_name="baselines/heuristic_full_obs", overrides=overrides)
        HydraConfig().cfg = c
        with open_dict(c):
            c.hydra = {"runtime": {"output_dir": str(out_dir)}}
        fix_config(c)
        c = setup_config(c, cfg.seed)
        prune_sensors(c)
        data = CollaborationDatasetV0(c.habitat.dataset)
        data = CollaborationDatasetV0(config=c.habitat.dataset, episodes=[e for e in data.episodes if e.episode_id in set(ids)])
        register_sensors(c)
        register_actions(c)
        register_measures(c)
        self.env = EnvironmentInterface(c, dataset=data, init_wg=False)
        self.env.initialize_perception_and_world_graph()
        self.agents = init_agents(c.evaluation.agents, self.env)
        self.batch = self.env.get_observations()
        self.zero = None

    # -- episode ------------------------------------------------------------------------------------
    @property
    def episode(self):
        return self.env.env.env.env._env.current_episode

    @property
    def sim(self):
        return self.env.sim

    def reset(self, episode_id: str) -> None:
        if self.episode.episode_id != episode_id:
            self.env.reset_environment(episode_id=episode_id)
        else:
            self.env.reset_environment(move_to_next_episode=False)
        self.batch = self.env.get_observations()
        self.names = {}
        if self.zero is None:  # a no-op of the joint action vector's shape, from one probe skill step
            fur = self.env.full_world_graph.get_all_furnitures()[0].name
            self.zero = np.zeros_like(np.asarray(self.skill(1, "Navigate", fur)[0]))
        for a in self.agents:
            a.reset()

    def skill(self, uid: int, tool: str, arg: str):
        """One step of a high-level skill: (low-level vector or None, response text; empty while running)."""
        return self.agents[uid].process_high_level_action(tool, arg, self.env.filter_obs_space(self.batch, uid))

    def step(self, actions: dict) -> dict:
        actions = {k: v for k, v in actions.items() if v is not None} or {1: self.zero}
        obs, _, _, info = self.env.step(actions)
        self.batch = self.env.parse_observations(obs)
        return info

    # -- state ---------------------------------------------------------------------------------------
    def satisfied_at(self) -> list[int]:
        m = self.env.env.env.env._env.task.measurements.measures["auto_eval_proposition_tracker"].get_metric()
        return list(m["proposition_satisfied_at"])

    def holding(self) -> list[bool]:
        """Whether each proposition holds now (PARTNR's satisfied_at only records the first time)."""
        m = self.env.env.env.env._env.task.measurements.measures["auto_eval_proposition_tracker"].get_metric()
        return [bool(r.is_satisfied) for r in m["state_sequence"][-1]]

    def head_look(self, target: np.ndarray | None, pitch: float = 15.0, max_yaw: float = 70.0) -> None:
        """Point the person's head camera at a world point (yaw clamped to max_yaw degrees), or straight ahead
        and `pitch` degrees down when target is None. PARTNR's camera always looks straight ahead."""
        import magnum as mn
        agent = self.sim.agents_mgr[1].articulated_agent
        cam = next(c for k, c in agent.params.cameras.items() if k.endswith("head"))  # keys carry the agent prefix
        o = np.array(cam.cam_offset_pos)
        if target is None:
            cam.cam_look_at_pos = mn.Vector3(*(o + [0.0, -0.5 * np.tan(np.radians(pitch)), 0.5]))
            return
        d = np.array(agent.sim_obj.transformation.inverted().transform_point(mn.Vector3(*target))) - o  # the camera's frame
        yaw = np.arctan2(d[0], d[2])
        lim = np.radians(max_yaw)
        if abs(yaw) > lim:  # the head turns only so far; keep the distance and the drop
            r = np.hypot(d[0], d[2])
            d[0], d[2] = r * np.sin(np.sign(yaw) * lim), r * np.cos(lim)
        cam.cam_look_at_pos = mn.Vector3(*(o + d))

    def percent_complete(self) -> float:
        return float(self.env.env.env.env._env.task.measurements.measures["task_percent_complete"].get_metric())

    def held(self, uid: int) -> str | None:
        g = self.sim.agents_mgr[uid].grasp_mgr
        if not g.is_grasped:
            return None
        h = self.sim.get_rigid_object_manager().get_object_handle_by_id(g.snap_idx)
        return self.env.perception.sim_handle_to_name.get(h, h)

    def name_of(self, panoptic_id: int) -> str | None:
        """Entity name of a panoptic value (rigid object id + 100), cached per episode."""
        if panoptic_id not in self.names:
            from habitat.sims.habitat_simulator import sim_utilities as sutils
            obj = sutils.get_obj_from_id(self.sim, int(panoptic_id) - 100) if panoptic_id >= 100 else None
            self.names[panoptic_id] = self.env.perception.sim_handle_to_name.get(obj.handle) if obj is not None else None
        return self.names[panoptic_id]

    def masks(self, key: str = "agent_1_head_panoptic", min_area: float = 2e-4) -> dict:
        """Visible entities in a panoptic camera: {name: (cx, cy, area)} in [0, 1] image coordinates."""
        pan = self.batch[key][0, ..., 0].cpu().numpy().astype(np.int64)
        H, W = pan.shape
        ids, inv, cnt = np.unique(pan.ravel(), return_inverse=True, return_counts=True)
        ys, xs = np.divmod(np.arange(pan.size), W)
        sx, sy = np.bincount(inv, xs), np.bincount(inv, ys)
        out = {}
        for k, v in enumerate(ids):
            name = self.name_of(int(v)) if v >= 100 else None
            if name and cnt[k] / pan.size >= min_area:
                out[name] = (float(sx[k] / cnt[k] / W), float(sy[k] / cnt[k] / H), float(cnt[k] / pan.size))
        return out

    def frame(self, key: str) -> np.ndarray:
        return self.batch[key][0, ..., :3].cpu().numpy().astype(np.uint8)
