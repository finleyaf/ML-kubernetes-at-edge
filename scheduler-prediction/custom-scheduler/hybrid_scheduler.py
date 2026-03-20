import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import os
import sys

PREDICTION_DIR = os.path.join(os.path.dirname(__file__), "..", "prediction")
sys.path.insert(0, os.path.abspath(PREDICTION_DIR))

from predictor import ClusterPredictor, FEATURES  # noqa: E402

ANOMALY_FEATURES = ["cpu_user", "cpu_system", "ram_used", "net_received", "net_sent"]


@dataclass
class NodeScore:
    node: str
    total_score: float
    predicted_load: float
    anomaly_risk: float


class NodeAnomalyMonitor:
    """Simple online anomaly monitor using rolling z-scores.

    It does not require labels at runtime and estimates anomaly risk from
    deviation against each node's own recent history.
    """

    def __init__(self, history_size: int = 30, z_threshold: float = 2.5):
        self.history_size = history_size
        self.z_threshold = z_threshold
        self.history: deque = deque(maxlen=history_size)

    def update(self, observation: Dict[str, float]) -> None:
        self.history.append([observation[f] for f in ANOMALY_FEATURES])

    def ready(self) -> bool:
        return len(self.history) >= max(10, self.history_size // 2)

    def risk(self, observation: Dict[str, float]) -> float:
        if not self.ready():
            return 0.0

        arr = np.array(self.history, dtype=float)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std < 1e-6, 1e-6, std)

        x = np.array([observation[f] for f in ANOMALY_FEATURES], dtype=float)
        z = np.abs((x - mean) / std)

        # Convert "largest z-score" into [0, 1] risk via logistic mapping.
        max_z = float(z.max())
        risk = 1.0 / (1.0 + math.exp(-(max_z - self.z_threshold)))
        return round(risk, 4)


class HybridScheduler:
    """Node ranking engine that combines prediction and anomaly awareness."""

    def __init__(
        self,
        model_dir: str,
        nodes: List[str],
        window_size: int = 10,
        anomaly_history: int = 30,
        anomaly_z_threshold: float = 2.5,
        weight_prediction: float = 0.6,
        weight_anomaly: float = 0.4,
    ):
        if abs((weight_prediction + weight_anomaly) - 1.0) > 1e-6:
            raise ValueError("weight_prediction + weight_anomaly must equal 1.0")

        self.weight_prediction = weight_prediction
        self.weight_anomaly = weight_anomaly

        self.predictor = ClusterPredictor(model_dir=model_dir, window_size=window_size)
        self.monitors: Dict[str, NodeAnomalyMonitor] = {}

        for node in nodes:
            self.predictor.add_node(node)
            self.monitors[node] = NodeAnomalyMonitor(
                history_size=anomaly_history,
                z_threshold=anomaly_z_threshold,
            )

    def update(self, node: str, observation: Dict[str, float]) -> None:
        self.predictor.update(node, observation)
        if node in self.monitors:
            self.monitors[node].update(observation)

    def _fallback_load(self, observation: Dict[str, float]) -> float:
        """Load estimate used before the prediction window is full.

        Uses a bounded weighted mix similar to predictor.predicted_load().
        """
        cpu = min(observation["cpu_user"] + observation["cpu_system"], 100.0) / 100.0
        mem = min(observation["ram_used"], 100.0) / 100.0
        net = min(observation["net_received"] + observation["net_sent"], 100.0) / 100.0
        return round(0.4 * cpu + 0.4 * mem + 0.2 * net, 4)

    def score_nodes(self, observations_by_node: Dict[str, Dict[str, float]]) -> List[NodeScore]:
        """Return all nodes sorted from best (lowest score) to worst."""
        scored: List[NodeScore] = []

        predictions = self.predictor.predict_all()

        for node, obs in observations_by_node.items():
            if node not in self.monitors:
                continue

            pred_load: Optional[float] = None
            if node in predictions:
                pred_load = predictions[node]["load_score"]
            if pred_load is None:
                pred_load = self._fallback_load(obs)

            anomaly_risk = self.monitors[node].risk(obs)
            total = self.weight_prediction * pred_load + self.weight_anomaly * anomaly_risk

            scored.append(
                NodeScore(
                    node=node,
                    total_score=round(float(total), 4),
                    predicted_load=round(float(pred_load), 4),
                    anomaly_risk=round(float(anomaly_risk), 4),
                )
            )

        scored.sort(key=lambda s: s.total_score)
        return scored

    def choose_node(self, observations_by_node: Dict[str, Dict[str, float]]) -> Optional[NodeScore]:
        ranked = self.score_nodes(observations_by_node)
        return ranked[0] if ranked else None


class PredictionOnlyScheduler(HybridScheduler):
    """Baseline ranking using prediction only (no anomaly term)."""

    def __init__(self, model_dir: str, nodes: List[str], window_size: int = 10):
        super().__init__(
            model_dir=model_dir,
            nodes=nodes,
            window_size=window_size,
            anomaly_history=30,
            weight_prediction=1.0,
            weight_anomaly=0.0,
        )
