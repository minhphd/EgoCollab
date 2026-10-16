# The collaboration env (helper/collab)

A person and a Stretch share a pick-and-place task in a BEHAVIOR-1K home (OmniGibson, Isaac Sim 5.1, the lab's RTX
5090). The env follows the Gymnasium API; the robot is the agent, the person is part of the environment.

    from helper.collab.env import CollabEnv, SkillEnv
    env = SkillEnv(CollabEnv("lunch"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step({"skill": "pick", "object": "mug_1", "step": 2})

| File | What it holds |
|---|---|
| env.py | `CollabEnv` (one frame, 1/15 s, per step; base speeds, joint goals, gripper, speech) and `SkillEnv` (navigate, pick, place, open, close) |
| world.py | the home: objects, placement, relations, what the person sees (rays from the eye), the floor map |
| stretch_sim.py | the simulated Stretch under the skills |
| ../stretch/skills.py | navigate, pick and place as controllers from the robot's state to commands, the same on the real robot |
| person.py | the scripted person and their gaze |
| bip.py | all of BIP, each piece under its equation |
| agents.py | the robot's policies: oracle, BIP, BIP with announce, ask or tell |
| tasks.py | the designed tasks |
| run.py, score.py | episodes, streams and clips; metrics against the person alone |

## Observation, action, information

- The robot observes its own state, the person's position, object positions (the simulator stands in for perception),
  and events: the person's actions as a recognizer reads them (the object right 70% of the time, else another object in
  view), their gaze fixations, what they say, and the robot's own steps done. Reward is -dt.
- `info` is privileged (the true steps done and who did them, the person's step): only the oracle and scoring read it.

## The Stretch

- Speeds: the Stretch 3's fastest (Hello Robot's stretch_body, robot_params_SE3.py "max"): base 0.3 m/s, lift
  0.15 m/s, arm 0.4 m/s, wrist 3 rad/s. A joint goal holds until reached; a gripper goal, until the fingers arrive
  (closing: until they hold something or meet).
- Trips: the robot watches the object it goes for; when it moves 10 cm before the robot has it, the person has taken
  it, and the robot stops, tucks its arm and leaves that step for 15 s.
- Navigation: A* on the scene's 1 cm floor map (10 cm grid; 0.22 m from obstacles, 0.45 m preferred; things a task
  stands on the floor are added to it), followed by pure pursuit 0.3 m ahead; turns in place when 0.5 rad off; 0.3 m/s
  and 0.8 rad/s; stops for the person within 0.5 m ahead (4 s at most). In the simulator the base follows the
  commanded speeds exactly (OmniGibson's Stretch drags its caster).
- Wrist and gripper: the robot drives with the wrist down, the gripper over its base. A reach keeps the wrist down
  when the lift (1.1 m) can carry the fingertips over what is in the way, else turns it level (fingertips 0.23 m
  higher, 0.28 m farther out); the wrist turns only at full lift, so the swinging gripper clears the furniture. Both
  fingers go to one angle: open (27 cm between the tips), ready (17 cm), closed (the tips meet at the point the skills
  aim at). OmniGibson's default gripper control swings one finger across and flings light objects.
- Pick: a base pose on reachable floor from which the arm's line passes over the object, not through a wall (a ray
  just over the object and one at 1.6 m, both stopped), with nothing else under the gripper's way down; a turn in
  place to aim; up to the travel height (over the object and whatever lies under the arm's way, the arm under any
  shelf above); out with the gripper ready; down until the fingertips are at the object's middle (wrist down; its
  upper 12 cm if it is tall) or at most 5 cm over its bottom (level), never under 2.5 cm, since closing fingertips
  swing 2.1 cm lower; close; up; in. The fingers close gently (0.4 rad/s). No grasp or motion planner: the lift and
  telescoping arm make the reach closed form. The grasp is OmniGibson's sticky grasp, the object attaching once a
  finger has touched it for 1/60 s while closing (OmniGibson's 0.3 s lets a light object bounce off the closing
  finger, each bounce starting the count again); only the target, only while the pick closes (armed at any other time,
  it takes whatever the gripper brushes); the fingers then hold where they met it. A pick that closes on nothing
  opens, rises, draws the arm in and stows before it reports, and the robot never drives with its arm out. What the
  robot holds passes through things until it lets go, as what the person holds does; held level, it rides with the
  gripper turned forward at full lift.
- Place: on a surface, the nearest free spot the arm reaches, not through a wall; into an open container, where the
  object fits, middle first; the object carried over whatever is in the way, the arm under any shelf above; down until
  the object is 2 cm over the surface (over the rim, for an open container, so it drops in), open, up, in. An object
  it cannot place goes back onto the furniture it came off; one it holds though the pick's last move failed goes on to
  its target.
- The robot takes only steps whose object fits its gripper (12 cm or less across its narrow side, 30 cm or less tall).

## The person

PARTNR's scripted person, calibrated on its people: the next step by distance and instruction order, away from where
the robot was last seen; a pause before each step; beliefs about where things are and what is done from what they see
and hear; a step the robot is seen on is left to it (2 s after noticing, 90% of the time), at once if the robot said
so; an object gone means the robot has it. Gaze follows HD-EPIC statistics.

## BIP as deployed

Gemini proposes the task's orderings from its title (once per task, cached) and revises them on surprise; actions are
read at 70%; completions are inferred (leaving a step after a placing action means it is done 95% of the time, else
50%). Every equation is in bip.py.

## Tasks

- lunch: a snack for two at the dining table; the small bowl before the apple in it. The bowl (14 cm across) is the
  person's: the gripper does not hold it.
- tidy: the child's room; the small box (a 29 x 31 cm storage box) from the bed onto the corridor shelf before two
  figures in it; four figures into the toy box. The figures for the small box are the person's: on the shelf it takes a
  level wrist, and these figures slip from a level grasp.
- groceries: from the entryway; the milk into the fridge before the cereal and the tea onto the counter; the fridge
  steps are the person's (the robot's wrist-down gripper has no room above it inside the fridge).

The robot takes only steps it can do: objects 12 cm or less across and 30 cm or less tall, placed on open surfaces or
into open containers, and not marked as the person's in the task.

## Running and tests

    runs/collab/run_collab.sh <out_dir> lunch alone,oracle,bip,bip+announce,bip+ask,bip+tell 0,1,2 clip=1
    python -m helper.collab.score <out_dir>
    python -m pytest tests -q          # BIP, the skills and the metrics, no simulator

One process per task. A simulator abort ends the process; the script starts it again, and the episode it died in is
recorded as crashed and scored apart. Clips show the person (with the robot when it is within 3 m) beside the robot's
arm, each camera on open floor with clear lines to what it films.

Metrics, against the person alone on the same task and seed: time relative to alone; robot trips for a step the person
had done or was doing when the trip began, or did during it, unless the trip did it (the person left it to the robot);
person trips for a step the robot had done or was doing, or did during it, likewise; utterances.
