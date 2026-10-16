"""helper/collab/score.py's metrics on hand-made episodes.

    python -m pytest tests/test_score.py -q
"""

from helper.collab.score import score, trips, waste


def test_waste_counts_trips_for_steps_the_other_did_or_was_doing():
    robot = [[0.0, 10.0, 1], [12.0, 20.0, 2], [21.0, 30.0, 3]]  # [start, end, step]
    person = [[5.0, 25.0, 3], [0.0, 11.0, 2]]
    done = [[11.0, 2, "person"], [26.0, 3, "person"]]  # [t, step, who]
    # step 2: the person finished it at 11 s, before the robot's trip at 12 s; step 3: the person was on it at 21 s
    assert waste(robot, person, done, "person") == (2, 17.0)


def test_a_trip_the_other_yielded_is_not_waste():
    robot, person = [[1.0, 77.0, 5]], [[0.5, 3.0, 5]]  # the person was on step 5, then left it to the robot
    assert waste(robot, person, [[62.0, 5, "robot"]], "person") == (0, 0.0)


def test_score_against_alone():
    alone = {"seconds": 100.0, "finished": True}
    st = {"policy": "bip", "seed": 0, "seconds": 70.0, "finished": True, "done": [], "robot_segments": [], "by": {"0": "robot"},
          "person_segments": [], "utterances": [{"t": 1.0, "who": "robot", "text": "I'll put it."}]}
    r = score(st, alone)
    assert r["time_rel"] == 0.7 and r["robot_utt"] == 1 and r["person_utt"] == 0 and r["robot_steps"] == 1
    assert score(dict(st, finished=False), alone)["time_rel"] is None


def test_a_trip_open_when_the_episode_ends_counts():
    st = {"robot_segments": [[1.0, 20.0, 2]], "robot_log": ["1.0 start 2", "20.0 failed 2", "21.0 start 4"], "seconds": 33.0}
    assert trips(st) == [[1.0, 20.0, 2], [21.0, 33.0, 4]]


def test_a_step_the_other_does_during_the_trip_wastes_it():
    robot, person = [[1.0, 29.9, 3]], [[0.0, 6.0, 2], [10.1, 15.0, 3]]  # the person was on step 2, then did step 3
    assert waste(robot, person, [[15.0, 3, "person"]], "person") == (1, 28.9)
