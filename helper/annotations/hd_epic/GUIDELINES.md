# Post-hoc task graph annotation

Each file in this folder is a task graph built after the fact from HD-EPIC narrations.
The schema is in `helper/datasets/posthoc_graph.py`. Numbered narrations come from
`python -m helper.datasets.hd_epic export-actions` (written to `data/hd_epic/actions/`).

## Nodes

- A node is a subgoal: a state of the world that the person brings about, such as
  "Capsule in coffee machine", "Milk frothed" or "Onion diced in pan".
- Label it as the resulting state, in plain words, at most about 6 words.
- Prefer pick-and-place style states (object at location, or object in a new state).
- Keep subgoals coarse: one coherent activity with one outcome, usually 15 to 100 narrations
  and about 5 to 12 subgoals per video. The same operation on several objects is one subgoal
  ("Dishes washed", not "Mug washed" and "Plate washed"); small setup steps join the step they
  serve ("Board out" belongs to the chopping); consecutive stages of one process are one
  subgoal ("Pepper toasted in butter", not butter in pan, pan heating, butter melted).
- A node covers the actions that achieve it. Fetching the tools
  and ingredients a subgoal needs belongs to that subgoal unless the fetch serves several.
- A subgoal done in several bouts lists all its action ranges. A node's actions may span videos.
- Actions that serve no subgoal (adjusting glasses, idle checks, reading the recipe) stay unassigned.
- Side activities in the same video (washing up, a second recipe) get their own group and nodes.
- `group` names a sub-task (for example "Coffee", "Sauce", "Cleanup"); the viewer lays out
  each group separately, so keep groups to about 2 to 6 per graph.

## Edges

- An edge u to v means v cannot be achieved, or would be pointless, unless u holds first.
  Physical access (a door is open, an object is in hand), ingredients (the onion is chopped
  before it is fried), and state (the pan is hot before the eggs go in) are all valid reasons.
- The order in which the person happened to act is not a reason by itself.
- Reusing one shared tool (a single colander, board or bowl) is not a reason either: a
  second tool or a second agent would remove that order. Edges describe the task, not
  one person's resources.
- Keep only direct preconditions; leave out edges implied by a path.
- Every edge has a one-line `why`.
- Edges may cross groups.
- The graph must be acyclic.

## Check

`python -m helper.datasets.hd_epic check` reports unknown actions, actions in two nodes and
cycles, and how many actions are assigned. Then run `build` and look at the episode in the viewer.

## Primitive actions

Below the subgoals sits a graph of primitive actions, stored under `"primitives"` in the task file:

```jsonc
"primitives": {
  "P01-20240202-110250": {
    "1": {"label": "open upper cupboard", "after": []},
    "3": {"label": "pick up mug", "after": [1]},
    "7": {"label": "place mug under nozzle", "after": [3, 6]}
  }
}
```

- Every narration listed in a subgoal's `actions` is one primitive, keyed by its index.
  Unassigned narrations are not primitives.
- `label` (optional): lowercase verb, object and, when it matters, the place: "pick up mug",
  "pour milk into frother", "close fridge". At most 6 words. Without it, the build uses the
  narration's `main_actions` pairs.
- `after`: the primitives that must happen first for this one to be possible or meaningful.
  An integer refers to the same video; `"<video_id>-<index>"` refers to another video.
  Valid reasons: the object is in hand (pick up before place), the container is open
  (open before take out), the ingredient is prepared, the device is on.
- Not valid: plain observed order, reuse of one shared tool, or a free hand. A repeated
  action ("continue stirring") depends on the first one; a check-only action
  ("look at the mug") depends on having the object, and nothing needs to depend on it.
- Keep only direct preconditions. Edges may cross subgoals, and then the source subgoal must
  reach the target subgoal in the subgoal graph; `check` reports each edge that breaks this.
