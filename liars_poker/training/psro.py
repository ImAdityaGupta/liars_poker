from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from liars_poker.core import GameSpec, ARTIFACTS_ROOT
from liars_poker.env import Env, Rules
from liars_poker.eval.match import play_match
from liars_poker.policies.base import Policy
from liars_poker.policies.random import RandomPolicy
from liars_poker.policies.population_mixture import PopulationMixturePolicy
from liars_poker.serialization import save_policy
from liars_poker.algo.psro import PSROState, nash_solver


@dataclass(frozen=True)
class SampleRequest:
    i: int
    j: int
    episodes: int


def default_sampling_plan(
    n_first: int,
    n_second: int,
    new_first_idx: int,
    new_second_idx: int,
    episodes_per_entry: int,
) -> List[SampleRequest]:
    if episodes_per_entry <= 0:
        return []

    requests: List[SampleRequest] = []
    for j in range(n_second):
        requests.append(SampleRequest(new_first_idx, j, episodes_per_entry))
    for i in range(n_first):
        if i == new_first_idx:
            continue
        requests.append(SampleRequest(i, new_second_idx, episodes_per_entry))
    return requests


def _sanitize_log(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_log(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_log(v) for v in obj]
    return str(obj)


@dataclass
class PopulationTracker:
    SIGMA_TOL = 1e-12
    birth_iter: Dict[int, int] = field(default_factory=dict)
    last_active_iter: Dict[int, int] = field(default_factory=dict)
    last_sigma: Dict[int, float] = field(default_factory=dict)

    def register(self, policy_id: int, iter_idx: int) -> None:
        if policy_id in self.birth_iter:
            return
        self.birth_iter[policy_id] = iter_idx
        self.last_active_iter[policy_id] = iter_idx
        self.last_sigma[policy_id] = 0.0

    def update(self, current_ids: List[int], sigma: np.ndarray, iter_idx: int) -> None:
        if len(current_ids) != len(sigma):
            raise ValueError("Tracker update: ids/sigma length mismatch.")
        sigma_tol = self.SIGMA_TOL
        for pid, weight in zip(current_ids, sigma):
            w = float(weight)
            self.last_sigma[pid] = w
            if w > sigma_tol:
                self.last_active_iter[pid] = iter_idx

    def choose_keep_indices(self, current_ids: List[int], iter_idx: int, *, cfg: Dict[str, float]) -> List[int]:
        max_size = int(cfg["max_size"])
        min_size = int(cfg["min_size"])
        recent_keep = int(cfg["recent_keep"])
        inactive_window = int(cfg["inactive_window"])
        sigma_tol = self.SIGMA_TOL

        if max_size < min_size:
            raise ValueError("prune_cfg max_size must be >= min_size.")
        if len(current_ids) <= min_size:
            return list(range(len(current_ids)))

        birth_sorted = sorted(
            current_ids,
            key=lambda pid: self.birth_iter.get(pid, -1),
            reverse=True,
        )
        protected = set(birth_sorted[:recent_keep])

        drop_candidates = []
        for pid in current_ids:
            age = iter_idx - self.birth_iter.get(pid, iter_idx)
            inactive = iter_idx - self.last_active_iter.get(pid, iter_idx)
            sigma = self.last_sigma.get(pid, 0.0)
            print(f"pid: {pid}, age: {age}, inactive: {inactive}, sigma: {sigma}, window: {inactive_window}")
            if inactive >= inactive_window and sigma <= sigma_tol:
                if pid not in protected:
                    drop_candidates.append(pid)

        print(f'protected: {protected}')
        print(f'drop candidates: {drop_candidates}')

        keep_ids = list(current_ids)
        if drop_candidates:
            drop_candidates = sorted(
                drop_candidates,
                key=lambda pid: self.last_active_iter.get(pid, iter_idx),
            )
            for pid in drop_candidates:
                if len(keep_ids) <= min_size:
                    break
                keep_ids.remove(pid)

        keep_set = set(keep_ids)
        return [idx for idx, pid in enumerate(current_ids) if pid in keep_set]


def prune_state(state: PSROState, keep_first: List[int], keep_second: List[int]) -> None:
    state.pop_first = [state.pop_first[i] for i in keep_first]
    state.ids_first = [state.ids_first[i] for i in keep_first]
    state.pop_second = [state.pop_second[j] for j in keep_second]
    state.ids_second = [state.ids_second[j] for j in keep_second]

    state.n = state.n[np.ix_(keep_first, keep_second)]
    state.mean = state.mean[np.ix_(keep_first, keep_second)]
    state.M2 = state.M2[np.ix_(keep_first, keep_second)]

def _extract_exploitability(log: object) -> Optional[Dict[str, float]]:
    if not isinstance(log, dict):
        return None
    if not log.get("computes_exploitability"):
        return None
    computer = log.get("computer")
    if computer is None:
        return None
    try:
        p_first, p_second = computer.exploitability()
    except Exception:
        return None
    return {"p_first": float(p_first), "p_second": float(p_second)}


class DefaultBackend:
    def __init__(self, spec: GameSpec, seed: int = 0):
        self.spec = spec
        self.rng = random.Random(seed)
        self.env = Env(spec)

    def evaluate(
        self,
        pop_first: List[Policy],
        pop_second: List[Policy],
        requests: Iterable[SampleRequest],
    ) -> List[Tuple[int, int, int, int]]:
        results: List[Tuple[int, int, int, int]] = []
        for req in requests:
            wins = play_match(
                self.env,
                pop_first[req.i],
                pop_second[req.j],
                episodes=req.episodes,
                seed=self.rng.randint(0, 2_147_483_647),
            )
            results.append((req.i, req.j, wins["P1"], wins["P2"]))
        return results


class PSROCheckpointer:
    def __init__(self, run_dir: str | Path, *, save_every: int = 1):
        self.run_dir = Path(run_dir)
        self.save_every = max(1, int(save_every))
        self.pop_first_dir = self.run_dir / "pop_first"
        self.pop_second_dir = self.run_dir / "pop_second"
        self.pop_first_dir.mkdir(parents=True, exist_ok=True)
        self.pop_second_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs.jsonl").touch(exist_ok=True)

    def _policy_dir(self, role: str, policy_id: int) -> Path:
        folder = self.pop_first_dir if role == "first" else self.pop_second_dir
        return folder / f"{policy_id:04d}"

    def save_policy(self, policy: Policy, role: str, policy_id: int) -> None:
        policy_dir = self._policy_dir(role, policy_id)
        if policy_dir.exists():
            return
        save_policy(policy, str(policy_dir))

    def _spec_to_dict(self, spec: GameSpec) -> Dict[str, object]:
        return {
            "ranks": spec.ranks,
            "suits": spec.suits,
            "hand_size": spec.hand_size,
            "claim_kinds": list(spec.claim_kinds),
            "suit_symmetry": spec.suit_symmetry,
        }

    def _policy_record(self, role: str, idx: int, policy_id: int, policy: Policy) -> Dict[str, object]:
        kind = getattr(policy, "POLICY_KIND", policy.__class__.__name__)
        path = self._policy_dir(role, policy_id).relative_to(self.run_dir)
        return {"idx": idx, "id": policy_id, "role": role, "kind": kind, "path": str(path)}

    def save_state(self, state: PSROState, iteration: int, extra: Dict[str, object]) -> None:
        manifest = {
            "iteration": iteration,
            "spec": self._spec_to_dict(state.spec),
            "pop_first": [
                self._policy_record("first", i, state.ids_first[i], pol)
                for i, pol in enumerate(state.pop_first)
            ],
            "pop_second": [
                self._policy_record("second", i, state.ids_second[i], pol)
                for i, pol in enumerate(state.pop_second)
            ],
        }
        manifest.update(extra)
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        np.savez_compressed(
            self.run_dir / "payoff_stats.npz",
            n=state.n,
            mean=state.mean,
            M2=state.M2,
        )

    def append_log(self, record: Dict[str, object]) -> None:
        log_path = self.run_dir / "logs.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def psro_loop(
    spec: GameSpec,
    *,
    iterations: int,
    oracle_first: Callable[[GameSpec, Policy, bool], Tuple[Policy, Dict]],
    oracle_second: Callable[[GameSpec, Policy, bool], Tuple[Policy, Dict]],
    meta_solver_fn: Callable[..., Tuple[np.ndarray, np.ndarray, float]] = nash_solver,
    meta_solver_kwargs: Optional[Dict] = None,
    episodes_per_entry: int = 2000,
    initial_first: Optional[Policy] = None,
    initial_second: Optional[Policy] = None,
    backend: Optional[DefaultBackend] = None,
    sampling_plan: Optional[Callable[..., List[SampleRequest]]] = None,
    run_dir: Optional[str] = None,
    prune_cfg: Optional[Dict] = None,
    debug: bool = False,
) -> Tuple[PSROState, Dict]:
    if meta_solver_kwargs is None:
        meta_solver_kwargs = {}
    if initial_first is None:
        initial_first = RandomPolicy()
    if initial_second is None:
        initial_second = RandomPolicy()

    rules = Rules(spec)
    initial_first.bind_rules(rules)
    initial_second.bind_rules(rules)

    state = PSROState.create(spec, [initial_first], [initial_second])
    backend = backend or DefaultBackend(spec)
    sampling_plan = sampling_plan or default_sampling_plan

    checkpointer = PSROCheckpointer(
        run_dir or Path(ARTIFACTS_ROOT) / "psro_runs" / "default_run"
    ) if run_dir is not None else None

    if checkpointer is not None:
        checkpointer.save_policy(initial_first, "first", state.ids_first[0])
        checkpointer.save_policy(initial_second, "second", state.ids_second[0])

    logs: Dict[str, List[Dict]] = {"iterations": []}

    tracker_first = PopulationTracker()
    tracker_second = PopulationTracker()
    for pid in state.ids_first:
        tracker_first.register(pid, 0)
    for pid in state.ids_second:
        tracker_second.register(pid, 0)

    prune_defaults = {
        "max_size": 200,
        "min_size": 5,
        "recent_keep": 25,
        "inactive_window": 50,
        "prune_every": 10,
    }
    cfg = None
    if prune_cfg is not None:
        cfg = dict(prune_defaults)
        cfg.update(prune_cfg)

    for iter_idx in range(iterations):
        timing: Dict[str, float] = {}

        t0 = time.perf_counter()
        sigma_first, sigma_second, meta_value = meta_solver_fn(state.U_mean(), **meta_solver_kwargs)
        timing["meta_solve_s"] = time.perf_counter() - t0
        tracker_first.update(state.ids_first, sigma_first, iter_idx)
        tracker_second.update(state.ids_second, sigma_second, iter_idx)

        if cfg is not None and (iter_idx + 1) % int(cfg["prune_every"]) == 0:
            before_first = len(state.pop_first)
            before_second = len(state.pop_second)
            keep_first = tracker_first.choose_keep_indices(state.ids_first, iter_idx, cfg=cfg)
            keep_second = tracker_second.choose_keep_indices(state.ids_second, iter_idx, cfg=cfg)
            if len(keep_first) < before_first or len(keep_second) < before_second:
                if debug:
                    print(
                        f"Prune iter {iter_idx + 1}: first {before_first}->{len(keep_first)}; "
                        f"second {before_second}->{len(keep_second)}"
                    )
                prune_state(state, keep_first, keep_second)
                sigma_first, sigma_second, meta_value = meta_solver_fn(state.U_mean(), **meta_solver_kwargs)

        if debug:
            # print(f"Payoff matrix:\n {state.U_mean()}")
            print(f"s1: {sigma_first}")
            print(f"s2: {sigma_second}\n")

        t1 = time.perf_counter()
        opp_for_first = PopulationMixturePolicy(state.pop_second, sigma_second)
        opp_for_first.bind_rules(rules)
        new_first, log_first = oracle_first(spec, opp_for_first, debug)
        timing["oracle_first_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        opp_for_second = PopulationMixturePolicy(state.pop_first, sigma_first)
        opp_for_second.bind_rules(rules)
        new_second, log_second = oracle_second(spec, opp_for_second, debug)
        timing["oracle_second_s"] = time.perf_counter() - t2

        new_first.bind_rules(rules)
        new_second.bind_rules(rules)

        new_first_idx = state.add_first(new_first)
        new_second_idx = state.add_second(new_second)
        tracker_first.register(state.ids_first[new_first_idx], iter_idx)
        tracker_second.register(state.ids_second[new_second_idx], iter_idx)

        if checkpointer is not None:
            checkpointer.save_policy(new_first, "first", state.ids_first[new_first_idx])
            checkpointer.save_policy(new_second, "second", state.ids_second[new_second_idx])

        t3 = time.perf_counter()
        requests = sampling_plan(
            len(state.pop_first),
            len(state.pop_second),
            new_first_idx,
            new_second_idx,
            episodes_per_entry,
        )
        results = backend.evaluate(state.pop_first, state.pop_second, requests)
        for i, j, wins_first, wins_second in results:
            state.update_entry_counts(i, j, wins_first, wins_second)
        timing["eval_s"] = time.perf_counter() - t3

        pairs_sampled = len(requests)
        episodes_total = sum(r.episodes for r in requests)

        record = {
            "iter": iter_idx + 1,
            "pop_sizes": {"first": len(state.pop_first), "second": len(state.pop_second)},
            "meta_value": meta_value,
            "sampling": {"pairs_sampled": pairs_sampled, "episodes_total": episodes_total},
            "timing": timing,
            "oracle_first_log": _sanitize_log(log_first),
            "oracle_second_log": _sanitize_log(log_second),
        }
        oracle_first_exp = _extract_exploitability(log_first)
        if oracle_first_exp is not None:
            record["oracle_first_exploitability"] = oracle_first_exp
        oracle_second_exp = _extract_exploitability(log_second)
        if oracle_second_exp is not None:
            record["oracle_second_exploitability"] = oracle_second_exp
        logs["iterations"].append(record)

        if checkpointer is not None and (iter_idx + 1) % checkpointer.save_every == 0:
            t4 = time.perf_counter()
            extra = {
                "meta_solver": {
                    "name": getattr(meta_solver_fn, "__name__", "meta_solver"),
                    "kwargs": meta_solver_kwargs,
                },
                "episodes_per_entry": episodes_per_entry,
            }
            checkpointer.save_state(state, iter_idx + 1, extra)
            checkpointer.append_log(record)
            timing["checkpoint_s"] = time.perf_counter() - t4

        if debug:
            assert state.mean.shape == (len(state.pop_first), len(state.pop_second))
            assert len(state.ids_first) == len(state.pop_first)
            assert len(state.ids_second) == len(state.pop_second)
            assert len(set(state.ids_first)) == len(state.ids_first)
            assert len(set(state.ids_second)) == len(state.ids_second)

    summary = {"run_dir": str(run_dir) if run_dir is not None else None, "logs": logs}
    return state, summary
