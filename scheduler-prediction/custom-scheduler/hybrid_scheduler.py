import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    base_predicted_load: float
    anomaly_risk: float
    weight_prediction: float
    weight_anomaly: float
    cpu_request_fraction: float
    memory_request_fraction: float
    capacity_penalty: float
    prediction_source: str


@dataclass
class NodeCapacity:
    cpu_millicores: float
    memory_mib: float


@dataclass
class WorkloadDemand:
    cpu_millicores: float = 0.0
    memory_mib: float = 0.0


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


class KMeansAnomalyMonitor:
    """Lightweight 1D k-means anomaly monitor across resource features.

    For each feature, two centroids are fit from recent history and a per-feature
    threshold is estimated as mean + n*std of nearest-centroid distances.
    """

    def __init__(self, history_size: int = 30, k: int = 2, threshold_std: float = 2.0):
        self.history_size = history_size
        self.k = max(2, int(k))
        self.threshold_std = float(threshold_std)
        self.history: deque = deque(maxlen=history_size)
        self._centers: Dict[str, np.ndarray] = {}
        self._thresholds: Dict[str, float] = {}
        self._ready = False

    def update(self, observation: Dict[str, float]) -> None:
        self.history.append([observation[f] for f in ANOMALY_FEATURES])
        if len(self.history) >= max(10, self.history_size // 2) and not self._ready:
            self._fit()

    def ready(self) -> bool:
        return self._ready

    def _fit_1d_kmeans(self, values: np.ndarray) -> np.ndarray:
        lo = float(values.min())
        hi = float(values.max())
        if abs(hi - lo) < 1e-9:
            return np.array([lo, hi], dtype=float)

        centers = np.array([lo, hi], dtype=float)
        for _ in range(20):
            d0 = np.abs(values - centers[0])
            d1 = np.abs(values - centers[1])
            assign = d0 <= d1
            if assign.all() or (~assign).all():
                break
            c0 = float(values[assign].mean())
            c1 = float(values[~assign].mean())
            new_centers = np.array([c0, c1], dtype=float)
            if np.allclose(new_centers, centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers

        centers.sort()
        return centers

    def _fit(self) -> None:
        arr = np.array(self.history, dtype=float)
        centers: Dict[str, np.ndarray] = {}
        thresholds: Dict[str, float] = {}

        for idx, feat in enumerate(ANOMALY_FEATURES):
            vals = arr[:, idx]
            c = self._fit_1d_kmeans(vals)
            dists = np.minimum(np.abs(vals - c[0]), np.abs(vals - c[1]))
            thr = float(dists.mean() + self.threshold_std * dists.std())
            if thr < 1e-6:
                thr = 1e-6
            centers[feat] = c
            thresholds[feat] = thr

        self._centers = centers
        self._thresholds = thresholds
        self._ready = True

    def risk(self, observation: Dict[str, float]) -> float:
        if not self.ready():
            return 0.0

        feature_risks: List[float] = []
        for feat in ANOMALY_FEATURES:
            x = float(observation[feat])
            c = self._centers[feat]
            thr = self._thresholds[feat]
            d = min(abs(x - float(c[0])), abs(x - float(c[1])))
            ratio = d / max(thr, 1e-6)
            r = 1.0 / (1.0 + math.exp(-4.0 * (ratio - 1.0)))
            feature_risks.append(float(r))

        return round(float(max(feature_risks)), 4)


class NSAAnomalyMonitor:
    """Lightweight NSA-inspired detector using recent history as self space."""

    def __init__(
        self,
        history_size: int = 30,
        num_detectors: int = 120,
        radius: float = 0.9,
        random_seed: int = 0,
    ):
        self.history_size = history_size
        self.num_detectors = max(20, int(num_detectors))
        self.radius = float(radius)
        self.rng = np.random.default_rng(int(random_seed))
        self.history: deque = deque(maxlen=history_size)
        self._self_mean: Optional[np.ndarray] = None
        self._self_std: Optional[np.ndarray] = None
        self._detectors: Optional[np.ndarray] = None

    def update(self, observation: Dict[str, float]) -> None:
        self.history.append([observation[f] for f in ANOMALY_FEATURES])
        if len(self.history) >= max(10, self.history_size // 2) and self._detectors is None:
            self._fit()

    def ready(self) -> bool:
        return self._detectors is not None

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        if self._self_mean is None or self._self_std is None:
            return arr
        return (arr - self._self_mean) / self._self_std

    def _fit(self) -> None:
        self_arr = np.array(self.history, dtype=float)
        self_mean = self_arr.mean(axis=0)
        self_std = self_arr.std(axis=0)
        self_std = np.where(self_std < 1e-6, 1.0, self_std)

        self._self_mean = self_mean
        self._self_std = self_std

        self_norm = self._normalise(self_arr)
        mins = self_norm.min(axis=0) - 1.0
        maxs = self_norm.max(axis=0) + 1.0

        detectors: List[np.ndarray] = []
        tries = 0
        max_tries = self.num_detectors * 200

        while len(detectors) < self.num_detectors and tries < max_tries:
            tries += 1
            cand = self.rng.uniform(mins, maxs)
            d_self = np.linalg.norm(self_norm - cand, axis=1)
            if np.any(d_self < self.radius):
                continue
            detectors.append(cand)

        if not detectors:
            # Fallback keeps system running even with highly compact self space.
            detectors.append(self.rng.uniform(mins, maxs))

        self._detectors = np.array(detectors, dtype=float)

    def risk(self, observation: Dict[str, float]) -> float:
        if not self.ready() or self._detectors is None:
            return 0.0

        x = np.array([observation[f] for f in ANOMALY_FEATURES], dtype=float)
        x = self._normalise(x)
        d = np.linalg.norm(self._detectors - x, axis=1)
        min_d = float(d.min())
        # Clip exponent input to avoid overflow for very large detector distances.
        expo = max(-60.0, min(60.0, 5.0 * (min_d - self.radius)))
        r = 1.0 / (1.0 + math.exp(expo))
        return round(float(r), 4)


def build_anomaly_monitor(
    source: str,
    history_size: int,
    z_threshold: float,
    nsa_num_detectors: int,
    nsa_radius: float,
    kmeans_threshold_std: float,
) -> object:
    source_norm = (source or "zscore").strip().lower()
    if source_norm == "zscore":
        return NodeAnomalyMonitor(history_size=history_size, z_threshold=z_threshold)
    if source_norm == "nsa":
        return NSAAnomalyMonitor(
            history_size=history_size,
            num_detectors=nsa_num_detectors,
            radius=nsa_radius,
        )
    if source_norm == "kmeans":
        return KMeansAnomalyMonitor(
            history_size=history_size,
            k=2,
            threshold_std=kmeans_threshold_std,
        )
    raise ValueError(f"Unsupported anomaly source: {source}")


class HybridScheduler:
    """Node ranking engine that combines prediction and anomaly awareness."""

    def __init__(
        self,
        model_dir: str,
        nodes: List[str],
        window_size: int = 10,
        anomaly_history: int = 30,
        anomaly_z_threshold: float = 2.5,
        anomaly_source: str = "zscore",
        nsa_num_detectors: int = 120,
        nsa_radius: float = 0.9,
        kmeans_threshold_std: float = 2.0,
        weight_prediction: float = 0.6,
        weight_anomaly: float = 0.4,
        adaptive_weighting: bool = False,
        adaptive_risk_low: float = 0.2,
        adaptive_risk_high: float = 0.7,
        adaptive_max_shift: float = 0.35,
        adaptive_min_prediction_weight: float = 0.05,
        adaptive_max_prediction_weight: float = 0.95,
        node_capacities: Optional[Dict[str, NodeCapacity]] = None,
        capacity_penalty_factor: float = 1.0,
    ):
        if abs((weight_prediction + weight_anomaly) - 1.0) > 1e-6:
            raise ValueError("weight_prediction + weight_anomaly must equal 1.0")

        self.weight_prediction = weight_prediction
        self.weight_anomaly = weight_anomaly
        self.anomaly_source = anomaly_source
        self.adaptive_weighting = adaptive_weighting
        self.adaptive_risk_low = adaptive_risk_low
        self.adaptive_risk_high = adaptive_risk_high
        self.adaptive_max_shift = adaptive_max_shift
        self.adaptive_min_prediction_weight = adaptive_min_prediction_weight
        self.adaptive_max_prediction_weight = adaptive_max_prediction_weight
        self.capacity_penalty_factor = max(0.0, float(capacity_penalty_factor))
        self.node_capacities = node_capacities or {}

        if self.adaptive_risk_low >= self.adaptive_risk_high:
            raise ValueError("adaptive_risk_low must be < adaptive_risk_high")

        self.predictor = ClusterPredictor(model_dir=model_dir, window_size=window_size)
        self.monitors: Dict[str, NodeAnomalyMonitor] = {}

        for node in nodes:
            self.predictor.add_node(node)
            self.monitors[node] = build_anomaly_monitor(
                source=anomaly_source,
                history_size=anomaly_history,
                z_threshold=anomaly_z_threshold,
                nsa_num_detectors=nsa_num_detectors,
                nsa_radius=nsa_radius,
                kmeans_threshold_std=kmeans_threshold_std,
            )

    def _effective_prediction_weight(self, anomaly_risk: float) -> float:
        if not self.adaptive_weighting:
            return self.weight_prediction

        risk = max(0.0, min(1.0, float(anomaly_risk)))
        if risk <= self.adaptive_risk_low:
            shift = self.adaptive_max_shift
        elif risk >= self.adaptive_risk_high:
            shift = -self.adaptive_max_shift
        else:
            ratio = (risk - self.adaptive_risk_low) / (self.adaptive_risk_high - self.adaptive_risk_low)
            shift = self.adaptive_max_shift * (1.0 - 2.0 * ratio)

        pred_w = self.weight_prediction + shift
        pred_w = max(self.adaptive_min_prediction_weight, min(self.adaptive_max_prediction_weight, pred_w))
        return float(pred_w)

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

    def _capacity_penalty(
        self,
        node: str,
        workload_request: Optional[WorkloadDemand],
    ) -> tuple[float, float, float]:
        if workload_request is None:
            return 0.0, 0.0, 0.0

        capacity = self.node_capacities.get(node)
        if capacity is None:
            return 0.0, 0.0, 0.0

        cpu_fraction = 0.0
        memory_fraction = 0.0

        if capacity.cpu_millicores > 0:
            cpu_fraction = workload_request.cpu_millicores / capacity.cpu_millicores
        if capacity.memory_mib > 0:
            memory_fraction = workload_request.memory_mib / capacity.memory_mib

        cpu_fraction = max(0.0, float(cpu_fraction))
        memory_fraction = max(0.0, float(memory_fraction))
        penalty = self.capacity_penalty_factor * max(cpu_fraction, memory_fraction)
        return cpu_fraction, memory_fraction, penalty

    def score_nodes(
        self,
        observations_by_node: Dict[str, Dict[str, float]],
        workload_request: Optional[WorkloadDemand] = None,
    ) -> List[NodeScore]:
        """Return all nodes sorted from best (lowest score) to worst."""
        scored: List[NodeScore] = []

        predictions = self.predictor.predict_all()

        for node, obs in observations_by_node.items():
            if node not in self.monitors:
                continue

            pred_load: Optional[float] = None
            prediction_source = "model"
            if node in predictions:
                pred_load = predictions[node]["load_score"]
            if pred_load is None:
                pred_load = self._fallback_load(obs)
                prediction_source = "fallback"

            base_pred_load = round(float(pred_load), 4)
            cpu_request_fraction, memory_request_fraction, capacity_penalty = self._capacity_penalty(
                node,
                workload_request,
            )
            projected_load = base_pred_load + capacity_penalty

            anomaly_risk = self.monitors[node].risk(obs)
            eff_pred_w = self._effective_prediction_weight(anomaly_risk)
            eff_anom_w = 1.0 - eff_pred_w
            total = eff_pred_w * projected_load + eff_anom_w * anomaly_risk

            scored.append(
                NodeScore(
                    node=node,
                    total_score=round(float(total), 4),
                    predicted_load=round(float(projected_load), 4),
                    base_predicted_load=base_pred_load,
                    anomaly_risk=round(float(anomaly_risk), 4),
                    weight_prediction=round(float(eff_pred_w), 4),
                    weight_anomaly=round(float(eff_anom_w), 4),
                    cpu_request_fraction=round(float(cpu_request_fraction), 4),
                    memory_request_fraction=round(float(memory_request_fraction), 4),
                    capacity_penalty=round(float(capacity_penalty), 4),
                    prediction_source=prediction_source,
                )
            )

        scored.sort(key=lambda s: s.total_score)
        return scored

    def choose_node(
        self,
        observations_by_node: Dict[str, Dict[str, float]],
        workload_request: Optional[WorkloadDemand] = None,
    ) -> Optional[NodeScore]:
        ranked = self.score_nodes(observations_by_node, workload_request=workload_request)
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
