"""
================================================================================
REPOSITORY 2: amharic_nli
MODULE: utils/logger.py
Paper: "ANLI: A Multi-Domain Benchmark Corpus, Automated Quality Filtering,
        and Subword Methodologies for Amharic Natural Language Inference"
Authors: Tesfaye Bekele Kassa, Dr. Tesfa Tegegne Asfaw, Beyene Feleke Mekonnen
================================================================================

Centralized Training & Experiment Logger for Amharic NLI Neural Architectures.
Tracks training epochs, loss convergence, validation accuracy, Macro F1,
hardware metadata, model checkpoints, and training duration with JSON summaries.
================================================================================
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List


class TrainingLogger:
    """
    Centralized training configuration, metrics, artifacts, and duration tracker
    for ANLI supervised experiments and baseline evaluations.
    """

    def __init__(
        self,
        experiment_name: str = "anli_proposed_bilstm_attention",
        log_dir: str = "logs",
        hyperparameters: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.hyperparameters = hyperparameters or {}
        self.artifacts: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0

        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configures file and stream handlers for logging output."""
        self.logger = logging.getLogger(f"{self.experiment_name}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

            # File log handler
            log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Logs and records hyperparameter configuration."""
        self.hyperparameters.update(params)
        self.logger.info("=" * 70)
        self.logger.info(f"ANLI EXPERIMENT CONFIGURATION: {self.experiment_name}")
        self.logger.info("=" * 70)
        for k, v in self.hyperparameters.items():
            self.logger.info(f"  - {k}: {v}")
        self.logger.info("=" * 70)

    def start_timer(self):
        """Starts timing the training run."""
        self.start_time = time.time()
        self.logger.info(f"Training run initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_timer(self) -> float:
        """Stops the timer and records elapsed training time."""
        if self.start_time is not None:
            self.end_time = time.time()
            self.duration_seconds = self.end_time - self.start_time
            mins, secs = divmod(self.duration_seconds, 60)
            hrs, mins = divmod(mins, 60)
            self.logger.info(
                f"Training concluded. Total Duration: {int(hrs):02d}h {int(mins):02d}m {secs:05.2f}s "
                f"({self.duration_seconds:.2f} seconds total)"
            )
        return self.duration_seconds

    def log_epoch(self, epoch: int, train_loss: float, val_acc: float, val_f1: float, extra: Optional[Dict[str, Any]] = None):
        """Records and prints standard epoch results."""
        extra_data = extra or {}
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_acc_pct": round(val_acc, 2),
            "val_macro_f1": round(val_f1, 4),
            "timestamp": datetime.now().isoformat(),
            **extra_data
        }
        self.metrics_history.append(record)
        extra_str = f" | {', '.join([f'{k}={v}' for k, v in extra_data.items()])}" if extra_data else ""
        self.logger.info(
            f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro F1: {val_f1:.4f}{extra_str}"
        )

    def log_metric(self, step: int, metrics: Dict[str, Any]):
        """Logs custom metrics dictionary for a step."""
        record = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(record)
        metric_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        self.logger.info(f"[Step {step}] Metrics logged: {metric_str}")

    def log_artifact(self, artifact_type: str, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Tracks model checkpoint, exported ONNX, or evaluation report."""
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        size_mb = round(size_bytes / (1024 * 1024), 2)
        artifact_entry = {
            "type": artifact_type,
            "path": path,
            "size_mb": size_mb,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.artifacts.append(artifact_entry)
        self.logger.info(f"[CHECKPOINT SAVED] {artifact_type} -> {path} ({size_mb} MB)")

    def export_summary_json(self, output_filename: Optional[str] = None) -> str:
        """Exports full experiment run summary to a machine-readable JSON file."""
        if not output_filename:
            output_filename = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")

        best_acc = max([m.get("val_acc_pct", 0) for m in self.metrics_history], default=0.0)
        best_f1 = max([m.get("val_macro_f1", 0) for m in self.metrics_history], default=0.0)

        summary = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "best_validation_accuracy_pct": best_acc,
            "best_validation_macro_f1": best_f1,
            "hyperparameters": self.hyperparameters,
            "artifacts": self.artifacts,
            "metrics_history": self.metrics_history
        }

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported experiment summary JSON to: {output_filename}")
        return output_filename


if __name__ == "__main__":
    logger = TrainingLogger(experiment_name="proposed_bilstm_attn_test", log_dir="logs")
    logger.log_hyperparameters({
        "architecture": "TwoLayerBiLSTMAttentionNLI",
        "embedding_dim": 300,
        "hidden_dim": 256,
        "dropout": 0.35,
        "learning_rate": 7.5e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "dataset": "all_in_one_cleaned.jsonl"
    })
    logger.start_timer()
    time.sleep(0.1)  # Simulate run
    logger.log_epoch(1, train_loss=0.4821, val_acc=81.25, val_f1=0.812)
    logger.log_epoch(2, train_loss=0.3512, val_acc=85.06, val_f1=0.850)
    logger.stop_timer()
    logger.export_summary_json()
