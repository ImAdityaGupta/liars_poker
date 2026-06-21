from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from liars_poker.algo.br_fitted_return_action_conditioned import (
    ActionConditionedFittedReturnBRTrainer,
)
from liars_poker.algo.neural_fsp_gpu import (
    CompactStrategyReservoir,
    GPUFSPStrategyCollector,
)
from liars_poker.core import GameSpec
from liars_poker.policies.action_conditioned import (
    ActionConditionedPolicy,
    ActionConditionedQPolicy,
)


def _spec_from_json(spec_json: str) -> GameSpec:
    data = json.loads(spec_json)
    return GameSpec(
        ranks=int(data["ranks"]),
        suits=int(data["suits"]),
        hand_size=int(data["hand_size"]),
        claim_kinds=tuple(data["claim_kinds"]),
        suit_symmetry=bool(data["suit_symmetry"]),
    )


class NeuralFSPTrainer:
    """GPU-native outer-loop fictitious play with neural BRs and averaging."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        spec: GameSpec,
        *,
        average_state_hidden_sizes: Sequence[int] = (512, 512),
        average_action_hidden_sizes: Sequence[int] = (128, 128),
        average_embedding_dim: int = 256,
        strategy_buffer_capacity: int = 1_000_000,
        validation_buffer_capacity: int = 20_000,
        validation_fraction: float = 0.02,
        strategy_batch_size: int = 4096,
        strategy_train_steps: int = 100,
        strategy_learning_rate: float = 1e-3,
        br_kwargs: Dict[str, object] | None = None,
        device: str | torch.device | None = None,
        fused_optimizer: bool | None = None,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.average_state_hidden_sizes = tuple(
            int(size) for size in average_state_hidden_sizes
        )
        self.average_action_hidden_sizes = tuple(
            int(size) for size in average_action_hidden_sizes
        )
        self.average_embedding_dim = int(average_embedding_dim)
        self.strategy_batch_size = int(strategy_batch_size)
        self.strategy_train_steps = int(strategy_train_steps)
        self.strategy_learning_rate = float(strategy_learning_rate)
        self.validation_fraction = float(validation_fraction)
        self.fused_optimizer = (
            self.device.type == "cuda"
            if fused_optimizer is None
            else bool(fused_optimizer)
        )
        self.br_kwargs = dict(br_kwargs or {})
        self.br_kwargs.pop("device", None)
        self.br_kwargs.pop("seed", None)

        self.average = ActionConditionedPolicy(
            spec,
            state_hidden_sizes=self.average_state_hidden_sizes,
            action_hidden_sizes=self.average_action_hidden_sizes,
            embedding_dim=self.average_embedding_dim,
            device=self.device,
        )
        self.average_nets = [self.average.model_p1, self.average.model_p2]
        self.average_optimizers = [
            self._make_optimizer(model) for model in self.average_nets
        ]
        input_dim = self.average.encoder.input_dim
        self.strategy_buffers = [
            CompactStrategyReservoir(
                strategy_buffer_capacity,
                input_dim,
                self.device,
            )
            for _ in range(2)
        ]
        self.strategy_validation_buffers = [
            CompactStrategyReservoir(
                validation_buffer_capacity,
                input_dim,
                self.device,
            )
            for _ in range(2)
        ]
        self.collector = GPUFSPStrategyCollector(spec, self.device)
        self.legal_masks = self.collector.legal_masks
        self.iteration = 0
        self.initial_strategy_collected = False
        self.last_br_states: list[Dict[str, torch.Tensor]] | None = None
        self._last_br_policy: ActionConditionedQPolicy | None = None

    def _make_optimizer(
        self,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        kwargs: Dict[str, object] = {"lr": self.strategy_learning_rate}
        if self.fused_optimizer and self.device.type == "cuda":
            kwargs["fused"] = True
        try:
            return torch.optim.Adam(model.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return torch.optim.Adam(model.parameters(), **kwargs)

    def average_policy(self) -> ActionConditionedPolicy:
        policy = ActionConditionedPolicy(
            self.spec,
            state_hidden_sizes=self.average_state_hidden_sizes,
            action_hidden_sizes=self.average_action_hidden_sizes,
            embedding_dim=self.average_embedding_dim,
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.average_nets[0].state_dict())
        policy.model_p2.load_state_dict(self.average_nets[1].state_dict())
        return policy.eval()

    def best_response_policy(self) -> ActionConditionedQPolicy:
        if self._last_br_policy is not None:
            return self._last_br_policy
        if self.last_br_states is None:
            raise RuntimeError("No best response has been trained yet.")
        policy = ActionConditionedQPolicy(
            self.spec,
            state_hidden_sizes=tuple(
                self.br_kwargs.get("state_hidden_sizes", (512, 512))
            ),
            action_hidden_sizes=tuple(
                self.br_kwargs.get("action_hidden_sizes", (128, 128))
            ),
            embedding_dim=int(self.br_kwargs.get("embedding_dim", 256)),
            device=self.device,
        )
        policy.model_p1.load_state_dict(self.last_br_states[0])
        policy.model_p2.load_state_dict(self.last_br_states[1])
        self._last_br_policy = policy.eval()
        return self._last_br_policy

    def _collect_policy(
        self,
        policy: ActionConditionedPolicy | ActionConditionedQPolicy,
        episodes_per_role: int,
        batch_size: int,
    ) -> tuple[list[Dict[str, int]], float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        records = [
            self.collector.collect(
                policy,
                role,
                episodes_per_role,
                self.strategy_buffers[role],
                validation_buffer=self.strategy_validation_buffers[role],
                validation_fraction=self.validation_fraction,
                batch_size=batch_size,
                generator=self.generator,
            )
            for role in (0, 1)
        ]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return records, time.perf_counter() - start

    def collect_initial_strategy(
        self,
        *,
        episodes_per_role: int,
        batch_size: int,
    ) -> Dict[str, object]:
        if self.initial_strategy_collected:
            return {"records": [0, 0], "collect_s": 0.0}
        records, collect_s = self._collect_policy(
            self.average,
            episodes_per_role,
            batch_size,
        )
        self.initial_strategy_collected = True
        return {
            "records": [record["records"] for record in records],
            "peak_states": [record["peak_states"] for record in records],
            "collect_s": collect_s,
        }

    def _train_average_role(
        self,
        role: int,
        *,
        steps: int | None = None,
    ) -> float:
        buffer = self.strategy_buffers[role]
        if buffer.size == 0:
            return 0.0
        model = self.average_nets[role]
        optimizer = self.average_optimizers[role]
        losses = []
        model.train()
        for _ in range(
            self.strategy_train_steps if steps is None else int(steps)
        ):
            features, actions, legal_rows = buffer.sample(
                self.strategy_batch_size,
                self.generator,
            )
            legal = self.legal_masks.index_select(0, legal_rows)
            logits = model.score_all(features, self.average.action_features)
            logits = logits.masked_fill(~legal, -1e9)
            loss = F.cross_entropy(logits, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        model.cache_actions(self.average.action_features)
        return float(np.mean(losses))

    def train_average(
        self,
        *,
        steps: int | None = None,
    ) -> tuple[list[float], float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        losses = [
            self._train_average_role(role, steps=steps)
            for role in (0, 1)
        ]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return losses, time.perf_counter() - start

    def validation_metrics(self) -> list[Dict[str, float]]:
        metrics = []
        for role in (0, 1):
            buffer = self.strategy_validation_buffers[role]
            if buffer.size == 0:
                metrics.append({"records": 0})
                continue
            features = buffer.features[: buffer.size].float()
            actions = buffer.actions[: buffer.size].long()
            legal_rows = buffer.legal_rows[: buffer.size].long()
            legal = self.legal_masks.index_select(0, legal_rows)
            with torch.inference_mode():
                logits = self.average_nets[role].score_all_cached(features)
                logits = logits.masked_fill(~legal, -1e9)
                loss = F.cross_entropy(logits, actions)
                accuracy = (logits.argmax(dim=1) == actions).float().mean()
            metrics.append(
                {
                    "records": buffer.size,
                    "cross_entropy": float(loss),
                    "sampled_action_accuracy": float(accuracy),
                }
            )
        return metrics

    def _new_br_trainer(
        self,
        opponent: ActionConditionedPolicy,
    ) -> ActionConditionedFittedReturnBRTrainer:
        trainer = ActionConditionedFittedReturnBRTrainer(
            self.spec,
            opponent,
            device=self.device,
            seed=self.seed + 10_000 * (self.iteration + 1),
            **self.br_kwargs,
        )
        if self.last_br_states is not None:
            for model, state in zip(trainer.q_nets, self.last_br_states):
                model.load_state_dict(state)
                model.eval()
                model.cache_actions(trainer.action_features)
        return trainer

    def run_iteration(
        self,
        *,
        br_iterations: int = 100,
        br_episodes_per_role: int = 512,
        br_rollout_batch_size: int = 256,
        strategy_episodes_per_role: int = 512,
        strategy_collection_batch_size: int = 256,
        strategy_train_steps: int | None = None,
    ) -> Dict[str, object]:
        initial = self.collect_initial_strategy(
            episodes_per_role=strategy_episodes_per_role,
            batch_size=strategy_collection_batch_size,
        )
        frozen_average = self.average_policy()
        br_trainer = self._new_br_trainer(frozen_average)

        br_start = time.perf_counter()
        last_br_record = None
        for _ in range(int(br_iterations)):
            last_br_record = br_trainer.run_iteration(
                episodes_per_role=br_episodes_per_role,
                rollout_batch_size=br_rollout_batch_size,
            )
        br_s = time.perf_counter() - br_start
        br_policy = br_trainer.policy()
        self.last_br_states = [
            {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            for model in br_trainer.q_nets
        ]
        self._last_br_policy = br_policy

        strategy_records, strategy_collect_s = self._collect_policy(
            br_policy,
            strategy_episodes_per_role,
            strategy_collection_batch_size,
        )
        strategy_losses, strategy_fit_s = self.train_average(
            steps=strategy_train_steps,
        )
        self.iteration += 1

        return {
            "iter": self.iteration,
            "initial_strategy": initial,
            "br_iterations": int(br_iterations),
            "br_last_record": last_br_record,
            "br_s": br_s,
            "strategy_records": [
                record["records"] for record in strategy_records
            ],
            "strategy_peak_states": [
                record["peak_states"] for record in strategy_records
            ],
            "strategy_collect_s": strategy_collect_s,
            "strategy_loss": strategy_losses,
            "strategy_fit_s": strategy_fit_s,
            "strategy_buffer_sizes": [
                buffer.size for buffer in self.strategy_buffers
            ],
            "strategy_records_seen": [
                buffer.seen for buffer in self.strategy_buffers
            ],
            "validation": self.validation_metrics(),
        }

    def checkpoint_dict(self) -> Dict[str, object]:
        return {
            "version": self.CHECKPOINT_VERSION,
            "spec_json": self.spec.to_json(),
            "config": {
                "average_state_hidden_sizes": self.average_state_hidden_sizes,
                "average_action_hidden_sizes": self.average_action_hidden_sizes,
                "average_embedding_dim": self.average_embedding_dim,
                "strategy_buffer_capacity": self.strategy_buffers[0].capacity,
                "validation_buffer_capacity": self.strategy_validation_buffers[
                    0
                ].capacity,
                "validation_fraction": self.validation_fraction,
                "strategy_batch_size": self.strategy_batch_size,
                "strategy_train_steps": self.strategy_train_steps,
                "strategy_learning_rate": self.strategy_learning_rate,
                "br_kwargs": self.br_kwargs,
                "fused_optimizer": self.fused_optimizer,
                "seed": self.seed,
            },
            "iteration": self.iteration,
            "initial_strategy_collected": self.initial_strategy_collected,
            "average_nets": [
                model.state_dict() for model in self.average_nets
            ],
            "average_optimizers": [
                optimizer.state_dict()
                for optimizer in self.average_optimizers
            ],
            "strategy_buffers": [
                buffer.state_dict() for buffer in self.strategy_buffers
            ],
            "strategy_validation_buffers": [
                buffer.state_dict()
                for buffer in self.strategy_validation_buffers
            ],
            "last_br_states": self.last_br_states,
            "generator_state": self.generator.get_state().cpu(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_dict(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "NeuralFSPTrainer":
        state = torch.load(path, map_location=device, weights_only=False)
        trainer = cls(
            _spec_from_json(state["spec_json"]),
            device=device,
            **state["config"],
        )
        trainer.iteration = int(state["iteration"])
        trainer.initial_strategy_collected = bool(
            state["initial_strategy_collected"]
        )
        for model, model_state in zip(
            trainer.average_nets,
            state["average_nets"],
        ):
            model.load_state_dict(model_state)
            model.eval()
            model.cache_actions(trainer.average.action_features)
        for optimizer, optimizer_state in zip(
            trainer.average_optimizers,
            state["average_optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)
        trainer.strategy_buffers = [
            CompactStrategyReservoir.from_state_dict(
                buffer_state,
                device=trainer.device,
            )
            for buffer_state in state["strategy_buffers"]
        ]
        trainer.strategy_validation_buffers = [
            CompactStrategyReservoir.from_state_dict(
                buffer_state,
                device=trainer.device,
            )
            for buffer_state in state["strategy_validation_buffers"]
        ]
        trainer.last_br_states = state.get("last_br_states")
        trainer.generator.set_state(state["generator_state"].cpu())
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if (
            state.get("cuda_rng_state") is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state"]]
            )
        return trainer
