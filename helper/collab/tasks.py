"""The three collaboration tasks, designed in house_single_floor for a person and a Stretch: pick and place only, every
pick and place within the Stretch's reach (heights 0.1-1.1 m), each with orderings and a false-belief moment.

A task: title (Gemini reads it), rooms to load, objects (name, category, start: relation and furniture, and the point
on it the object starts nearest to), steps (an object, a relation and a target; `closed` when the target has a door),
and orderings (u, v): step v needs step u done. Starting points sit along the furniture's edge that faces open floor,
apart, so a gripper has room around each.

    task = TASKS["lunch"]; task["steps"][0]["obj"]
"""

from __future__ import annotations


def step(obj: str, rel: str, target: str, closed: bool = False, robot: bool = True) -> dict:
    """robot=False: a step the Stretch is not to take (its reach there does not hold, though the object fits)."""
    return {"obj": obj, "rel": rel, "target": target, "closed": closed, "robot": robot}


def task(title: str, rooms: list[str], objects: list[tuple], steps: list[dict], orderings: list[tuple],
         starts: tuple | None = None) -> dict:
    """objects: (name, category, model, relation, furniture[, (x, y) it starts nearest to]); starts: the person's and the
    robot's (x, y), when the open floor near the first step's object is not in the task's rooms."""
    nodes = [f"v{i}" for i in range(len(steps))]
    word = lambda n: n.rsplit("_", 1)[0].replace("_", " ")
    return {"title": title, "scene": "house_single_floor", "rooms": rooms,
            "objects": [{"name": o[0], "category": o[1], "model": o[2], "start": (o[3], o[4]), "near": o[5] if len(o) > 5 else None}
                        for o in objects],
            "steps": steps, "orderings": orderings, "nodes": nodes,
            "labels": [f"put the {word(s['obj'])} {'in' if s['rel'] == 'inside' else 'on'} the {word(s['target'])}" for s in steps],
            "step_objects": {v: [s["obj"], s["target"]] for v, s in zip(nodes, steps)}, "starts": starts}


KITCHEN_A, KITCHEN_B, TABLE = "countertop_kelker_0", "countertop_kelzer_0", "breakfast_table_rhjoby_0"

# A snack for two: the small bowl to the table and then the apple in it; mugs, cupcakes, an orange and a can to the
# table. Two mugs of one model; nothing starts on the cooktop (its grates stand over the counter and tilt what is half
# on them). The bowl (14 cm across) is the person's: the gripper does not hold it. False belief: the robot can move the bowl or take a step
# the person planned while they work at the other counter.
LUNCH = task(
    "Set out a snack for two at the dining table: put the small bowl on the table first, then the apple in it; and two "
    "mugs, two cupcakes, an orange and a can of soda on the table",
    ["kitchen_0", "dining_room_0", "living_room_0"],
    [("bowl_1", "bowl", "eipwho", "ontop", KITCHEN_B, (4.85, -1.72)), ("apple_1", "apple", "bwteqh", "ontop", KITCHEN_A, (4.38, -0.9)),
     ("mug_1", "mug", "kitxam", "ontop", KITCHEN_A, (4.38, -1.35)), ("mug_2", "mug", "kitxam", "ontop", KITCHEN_A, (4.38, -1.75)),
     ("cupcake_1", "cupcake", "pfwrlq", "ontop", KITCHEN_B, (5.55, -1.72)), ("cupcake_2", "cupcake", "fabdnw", "ontop", KITCHEN_B, (6.6, -1.72)),
     ("orange_1", "orange", "nlygru", "ontop", KITCHEN_A, (4.38, 0.15)), ("can_1", "can_of_soda", "evcxlr", "ontop", KITCHEN_B, (7.0, -1.72))],
    [step("bowl_1", "ontop", TABLE, robot=False), step("apple_1", "inside", "bowl_1"), step("mug_1", "ontop", TABLE),
     step("mug_2", "ontop", TABLE), step("cupcake_1", "ontop", TABLE), step("cupcake_2", "ontop", TABLE),
     step("orange_1", "ontop", TABLE), step("can_1", "ontop", TABLE)],
    [(0, 1)], starts=((5.3, -0.8), (6.6, -0.85)))

# Tidy the child's room: the small box (on the bed) onto the corridor shelf first, then two figures in it; four more
# figures from the bed, nightstand and desk into the toy box on the floor. Only the desk's north end is in reach from the
# floor (a chair stands before the rest). The toy box stands along the north wall, clear of the walkway by the bed. False belief: the robot boxes a toy the person comes back for.
# The small box is a 29 x 31 x 7 cm storage box: the figures are 15 to 17 cm long. The robot leaves the figures for it to
# the person: on the shelf, the box takes a level wrist, and these figures slip out of a level grasp.
BED, NIGHTSTAND, DESK, SHELF_LOW, SHELF = "bed_rrcvaq_1", "nightstand_wbxekb_2", "desk_bhkhxo_0", "shelf_amyper_1", "shelf_amyper_2"
TIDY = task(
    "Tidy the child's room: put the small box on the corridor shelf first, then two toy figures in it; the other four "
    "toy figures go in the toy box",
    ["childs_room_1", "corridor_0"],
    [("toy_box_1", "toy_box", "qgfbdj", "floor", (22.6, 14.83, 1.5708)), ("small_box_1", "storage_box", "eicujj", "ontop", BED, (20.6, 13.3)),
     ("toy_figure_1", "toy_figure", "fnueyn", "ontop", BED, (21.0, 13.45)), ("toy_figure_2", "toy_figure", "gnofgv", "ontop", BED, (21.6, 13.45)),
     ("toy_figure_3", "toy_figure", "jddrck", "ontop", NIGHTSTAND, (20.6, 13.9)), ("toy_figure_4", "toy_figure", "acymin", "ontop", DESK, (23.7, 13.95)),
     ("toy_figure_5", "toy_figure", "orunry", "ontop", SHELF_LOW, (20.05, 13.0)), ("toy_figure_6", "toy_figure", "zuntus", "ontop", BED, (22.1, 13.45))],
    [step("small_box_1", "ontop", SHELF), step("toy_figure_1", "inside", "small_box_1", robot=False),
     step("toy_figure_2", "inside", "toy_box_1"), step("toy_figure_3", "inside", "toy_box_1"),
     step("toy_figure_4", "inside", "toy_box_1"), step("toy_figure_5", "inside", "toy_box_1"),
     step("toy_figure_6", "inside", "small_box_1", robot=False)],
    [(0, 1), (0, 6)], starts=((21.9, 14.4), (23.3, 14.3)))

# Put away the groceries from the entryway: the milk into the fridge first, then the cereal and the tea on the counter;
# the can and the jam into the fridge, the lemon on the counter, the battery on the other entryway cabinet. The robot
# cannot reach into the fridge (no room above for its gripper): those steps are the person's. False belief: the person
# comes back to the entryway for something the robot already put away.
CAB_A, CAB_B, FRIDGE = "bottom_cabinet_rhdbzv_0", "bottom_cabinet_rhdbzv_1", "fridge_dszchb_0"
GROCERIES = task(
    "Put away the groceries from the entryway: first the milk into the fridge, then the cereal and the tea on the kitchen "
    "counter; the can and the jam go in the fridge, the lemon on the counter, and the battery on the other entryway "
    "cabinet",
    ["entryway_0", "kitchen_0"],
    [("milk_1", "carton_of_milk", "orikxq", "ontop", CAB_A, (0.93, -1.15)), ("can_1", "can_of_soda", "opivig", "ontop", CAB_A, (0.93, -1.45)),
     ("jam_1", "jar_of_jam", "vxjbjf", "ontop", CAB_B, (0.94, 0.15)), ("cereal_1", "box_of_cereal", "pfwgkm", "ontop", CAB_B, (0.94, 0.4)),
     ("tea_1", "tea_bag", "kmvcrb", "ontop", CAB_A, (0.93, -1.75)), ("lemon_1", "lemon", "bbvonm", "ontop", CAB_B, (0.94, 0.65)),
     ("battery_1", "battery", "dcjyzg", "ontop", CAB_B, (0.94, 0.88))],
    [step("milk_1", "inside", FRIDGE, closed=True), step("can_1", "inside", FRIDGE, closed=True),
     step("jam_1", "inside", FRIDGE, closed=True), step("cereal_1", "ontop", KITCHEN_A), step("tea_1", "ontop", KITCHEN_A),
     step("lemon_1", "ontop", KITCHEN_A), step("battery_1", "ontop", CAB_A)],
    [(0, 3), (0, 4)], starts=((0.0, 0.4), (-0.29, -0.45)))

TASKS = {"lunch": LUNCH, "tidy": TIDY, "groceries": GROCERIES}
