# Collaborative tasks in BEHAVIOR-1K

A person and a robot share a household rearrangement task in a BEHAVIOR-1K home (OmniGibson, Isaac Sim 5.1, the
lab's RTX 5090). Code: helper/behavior (collab_tasks.py, collab.py, collab_score.py, collab_clip.py, human.py,
floor.py, sim.py).

## Tasks

- The 2026 challenge's rearrangement activities with their sampled starting states (7 homes). A goal in BDDL is
  grounded on the instance's objects into steps (object, relation, target); a step whose target is itself moved comes
  after the move. 44 activities have 4 to 20 steps; setting the table has 8 (two plates on the table, a cupcake on each
  plate, a fork and a knife next to each).
- A step is done when its relation holds between the objects' boxes at their current poses for 0.5 s, or at once after
  a hand placed it; it is undone only once its object moves more than 5 cm. OmniGibson's own BDDL predicates (OnTop,
  Inside, NextTo, Under) reject some placements people accept on small targets such as plates; each episode records
  which steps they count as done at the end, beside ours.
- Placement turns the object to its flatter resting orientation, then finds a spot for that footprint by casting rays
  down through the target: its top for on; a shelf or the bottom for in (bookcases, cabinets, cups), lowest first;
  the base of a cabinet or the floor beneath for under; the level the target stands on for next to. Rays at the
  footprint's corners keep it clear of walls and rims, and a placement counts only if the object stays put. In an
  open box, basket or cup, an object rests on the floor or on what is already there and is held in place; one that
  does not fit lying stands upright, out of the rim if its middle is below it (a toothbrush in a cup). A thing in a
  hand passes through others and is solid again once set down.
- Every episode starts from the same saved state: the person's view, stride and the placement caches are reset.

## The person

- An Isaac Sim character (M_Medical_01) with motion-captured walk and idle clips retargeted onto its skeleton and
  skinned in numpy every frame (Isaac Sim's own skeletal animation aborts under OmniGibson's Fabric delegate). The eye
  camera sits just in front of the eye joints and turns and bends with the head.
- The right arm reaches by two-bone inverse kinematics; the torso bends up to 30 degrees for low targets; the neck turns
  toward what is handled. A pick is a 0.5 s reach, the object into the hand, and a 0.4 s return.
- PARTNR's scripted person: the nearest open step first, a short pause before each step (0.1 times the last step's
  work). What the person counts as done, and where they think each object is, comes from what they did or saw.
- The person minds the robot: they glance at it (it shares gaze with the next step's objects), leave objects within
  1.5 m of where they last saw it (each metre inside counts as a metre farther), leave a step the robot is seen on,
  and, finding an object gone, take it that the robot has it and leave that step for 20 s.
- Gaze: HD-EPIC-calibrated fixations on the visible entities (helper/partnr/gaze.py).

## The robot

- A Stretch, visual-only, walking the same planner at 0.8 m/s; its lift and telescoping arm extend to what it picks or
  places. It starts behind the person.
- Policies: oracle (knows the task and the person's current step), BIP (infers the person's current step from their
  actions, gaze fixations and completions; the step orderings given), and BIP with speech:
  - announce (delegation): "I'll put the plate on the breakfast table."
  - ask (clarification): "Shall I put the plate on the breakfast table?"; the person answers, and declines a step they
    are on.
  - tell (correction): when the robot picks an object up ("I have the oatmeal box.") and again once it is in place
    ("I put the oatmeal box in the top cabinet."), by value: the chance the person missed the move (from gaze dwell
    on the object, or on the robot carrying it) times the chance they go for it (while carried: certain if the
    person's observed action names it, else BIP's posterior over their current and next step; once in place: certain,
    as the person does every step they believe open), against the cost of hearing it. The robot uses only what it
    observes.
  - tell without gaze (the chance of missing it from time in view only) and always tell.

## Walking room

The scene's 1 cm floor map with doors open; a 10 cm planning grid whose cells keep the least clearance of their pixels
(OmniGibson's own grid loses thin railings, door lines and furniture edges). A* keeps bodies 0.22 m from obstacles and
prefers 0.45 m. The person has the right of way: the robot stops rather than come within 0.5 m of them (4 s at
most), and the person walks around a robot in the way if that adds at most 2.5 m, else through it; standing spots are 0.45 m off the edge of the furniture an object rests on, reachable, and away from
the other agent. helper/behavior/clearance.py checks every recorded position against the full-resolution map.

## Metrics (helper/behavior/collab_score.py)

Against the person alone on the same task and seed: time relative to alone; robot trips for a step the person had
done or was doing; person trips for a step the robot had done or was doing (count and seconds each); robot and person
utterances.
