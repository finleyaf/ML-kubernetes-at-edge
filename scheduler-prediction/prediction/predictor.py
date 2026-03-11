import numpy as np
import pickle
import os

FEATURES = ["cpu_user", "cpu_system", "ram_used", "net_received", "net_sent"]


class NodePredictor:
    """Lightweight predictor for a single node's resource utilisation.

    Maintains a sliding window of recent observations and predicts
    future resource values using a pre-trained linear regression model.
    """

    def __init__(self, model_path, scaler_path, window_size=10):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.window_size = window_size
        self.buffer = []

    def update(self, observation):
        """Add a new observation to the sliding window.

        Args:
            observation: dict with keys matching FEATURES
        """
        values = [observation[f] for f in FEATURES]
        self.buffer.append(values)

        # keep only the most recent window
        if len(self.buffer) > self.window_size:
            self.buffer = self.buffer[-self.window_size:]

    def ready(self):
        """Check if the buffer has enough data to make a prediction."""
        return len(self.buffer) >= self.window_size

    def predict(self):
        """Predict future resource utilisation.

        Returns:
            dict with predicted values for each feature (normalised 0-1),
            or None if buffer is not full.
        """
        if not self.ready():
            return None

        # scale and flatten the window
        window = np.array(self.buffer[-self.window_size:])
        scaled = self.scaler.transform(window)
        X = scaled.flatten().reshape(1, -1)

        # predict
        pred = self.model.predict(X)[0]

        # clip to [0, 1] range
        pred = np.clip(pred, 0, 1)

        return {f: float(pred[i]) for i, f in enumerate(FEATURES)}

    def predicted_load(self):
        """Get a single aggregate load score (0-1) for scheduling decisions.

        Combines predicted CPU and memory into a single score.
        """
        pred = self.predict()
        if pred is None:
            return None

        # weighted combination: CPU (user+system) and memory
        cpu_load = pred["cpu_user"] + pred["cpu_system"]
        mem_load = pred["ram_used"]
        net_load = pred["net_received"] + pred["net_sent"]

        # normalise to 0-1 (cpu can sum to 2 max, net can sum to 2 max)
        score = 0.4 * min(cpu_load, 1.0) + 0.4 * mem_load + 0.2 * min(net_load, 1.0)
        return round(score, 4)


class ClusterPredictor:
    """Manages predictors for all worker nodes in the cluster."""

    def __init__(self, model_dir, window_size=10):
        self.predictors = {}
        self.model_dir = model_dir
        self.window_size = window_size

    def add_node(self, node_name):
        """Load a trained model for a node."""
        node_short = node_name.replace("k3s-", "")
        model_path = os.path.join(self.model_dir, f"model_{node_short}.pkl")
        scaler_path = os.path.join(self.model_dir, f"scaler_{node_short}.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found for {node_name} at {model_path}")

        self.predictors[node_name] = NodePredictor(model_path, scaler_path, self.window_size)
        print(f"Loaded predictor for {node_name}")

    def update(self, node_name, observation):
        """Feed a new observation to a node's predictor."""
        if node_name in self.predictors:
            self.predictors[node_name].update(observation)

    def predict_all(self):
        """Get predictions for all nodes.

        Returns:
            dict of node_name -> prediction dict
        """
        results = {}
        for node, predictor in self.predictors.items():
            if predictor.ready():
                results[node] = {
                    "features": predictor.predict(),
                    "load_score": predictor.predicted_load()
                }
        return results

    def rank_nodes(self):
        """Rank nodes by predicted load (lowest load first).

        Returns:
            list of (node_name, load_score) tuples, sorted ascending
        """
        predictions = self.predict_all()
        ranked = [(node, pred["load_score"]) for node, pred in predictions.items()]
        ranked.sort(key=lambda x: x[1])
        return ranked
