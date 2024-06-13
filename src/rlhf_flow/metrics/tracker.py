from typing import List, Dict
import numpy as np

class AlignmentMetricsTracker:
    """Tracks reward margin growth, implicit accuracy, and KL divergence during DPO training."""
    def __init__(self):
        self.losses: List[float] = []
        self.margins: List[float] = []
        self.accuracies: List[float] = []

    def log_step(self, loss: float, margin: float, accuracy: float):
        self.losses.append(loss)
        self.margins.append(margin)
        self.accuracies.append(accuracy)

    def summary(self) -> Dict[str, float]:
        return {
            "mean_loss": float(np.mean(self.losses)) if self.losses else 0.0,
            "mean_margin": float(np.mean(self.margins)) if self.margins else 0.0,
            "win_rate": float(np.mean(self.accuracies)) if self.accuracies else 0.0,
            "steps_recorded": len(self.losses)
        }

# Margin histogram exporter

# Ring buffer optimization

# Rolling average window

# Accuracy confidence interval
# Alignment metrics tracker

# Reward margin histogram
