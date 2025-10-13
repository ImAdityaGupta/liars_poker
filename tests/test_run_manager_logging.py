import json
import os

from liars_poker.core import GameSpec
from liars_poker.infoset import InfoSet, NO_CLAIM
from liars_poker.io.run_manager import RunManager
from liars_poker.policies.commit_once import CommitOnceMixture
from liars_poker.policies.random import RandomPolicy
from liars_poker.policies.tabular import TabularPolicy


def test_run_manager_creates_and_indexes(tmp_path) -> None:
    spec = GameSpec(ranks=3, suits=1, hand_size=1)
    run = RunManager(spec, save_root=str(tmp_path), seed=99)

    rand = RandomPolicy()
    a0 = run.log_policy(
        rand,
        role="average",
        parents=[],
        mixing=None,
        seed=1,
        train={"algo": "init_random"},
    )
    assert a0 == "A0"

    tab = TabularPolicy()
    tab.set(InfoSet(pid=0, last_idx=NO_CLAIM, hand=(0,), history=()), {0: 1.0})
    b0 = run.log_policy(
        tab,
        role="best_response",
        parents=[{"id": a0, "role": "avg", "weight": 1.0}],
        mixing=None,
        seed=2,
        train={"algo": "best_response"},
    )
    assert b0 == "B0"

    mix = CommitOnceMixture([rand, tab], [0.9, 0.1])
    a1 = run.log_policy(
        mix,
        role="average",
        parents=[{"id": a0, "role": "avg", "weight": 0.9}, {"id": b0, "role": "br", "weight": 0.1}],
        mixing={"impl": "commit_once", "eta": 0.1},
        seed=3,
        train={"algo": "fsp_average"},
    )
    assert a1 == "A1"
    assert run.current_policy_id() == a1

    averages = run.list_policies("average")
    assert averages == ["A0", "A1"]

    lineage = run.expand_lineage(a1)
    ids = {parent_id for parent_id, _, _ in lineage}
    assert {a0, b0}.issubset(ids)

    run.log_event("unit.test", iter=0)
    timeline = os.path.join(run.trainer_dir, "timeline.jsonl")
    assert os.path.exists(timeline)
    with open(timeline, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert any(event["event"] == "unit.test" for event in lines)

    run.write_iteration_artifacts(0, {"score": 1}, {"summary": True})
    iter_dir = os.path.join(run.trainer_dir, "iterations", "0000")
    assert os.path.exists(os.path.join(iter_dir, "eval.json"))

