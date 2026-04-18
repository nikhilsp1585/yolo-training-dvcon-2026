"""
task_encoder.py
───────────────
DistilBERT-based task encoder for semantic embedding of task descriptions.

Responsibilities:
  1. Load a (optionally pruned) DistilBERT model
  2. Encode task description strings → fixed-size task embedding vectors
  3. Cross-modal fusion: filter/re-score YOLO detections using task relevance
  4. Measure encoding latency per task

Usage:
    from task_encoder import TaskEncoder
    encoder = TaskEncoder()
    embedding = encoder.encode("I want to cook a meal")
    filtered = encoder.filter_detections(detections, task_key="T01_cooking")
"""

import time
import torch
import numpy as np
from typing import Optional
from transformers import DistilBertTokenizer, DistilBertModel
from coco_tasks import COCO_TASKS, COCO_CLASSES


class TaskEncoder:
    """
    Lightweight DistilBERT encoder that:
      - Converts text task descriptions into 768-d embeddings
      - Performs task-guided detection filtering (cross-modal fusion simulation)
      - Reports encoding latency for edge deployment estimation
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 device: str = "auto", max_length: int = 64):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[TaskEncoder] Loading {model_name} on {self.device}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

        # Cache embeddings for the 14 predefined tasks (avoid re-encoding)
        self._task_embedding_cache: dict[str, torch.Tensor] = {}
        print(f"[TaskEncoder] Ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a text string into a 768-d CLS embedding.

        Args:
            text: task description string

        Returns:
            Tensor of shape [768] (CLS token embedding, L2-normalised)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token = first token of last hidden state
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)

        # L2 normalise for stable cosine similarity later
        cls_embedding = cls_embedding / (cls_embedding.norm() + 1e-8)

        return cls_embedding.cpu()

    def encode_task(self, task_key: str) -> torch.Tensor:
        """Encode a predefined task by its key, with caching."""
        if task_key not in self._task_embedding_cache:
            desc = COCO_TASKS[task_key]["description"]
            self._task_embedding_cache[task_key] = self.encode(desc)
        return self._task_embedding_cache[task_key]

    def encode_all_tasks(self) -> dict[str, torch.Tensor]:
        """Pre-encode all 14 task embeddings and cache them."""
        print("[TaskEncoder] Pre-encoding all 14 task embeddings...")
        for task_key in COCO_TASKS:
            self.encode_task(task_key)
        print("[TaskEncoder] Done.")
        return self._task_embedding_cache

    # ─────────────────────────────────────────────────────────────────────────
    # Latency measurement
    # ─────────────────────────────────────────────────────────────────────────

    def measure_encoding_latency(self, task_key: str, n_runs: int = 20) -> dict:
        """
        Measure encoding latency over n_runs to get stable timing.

        Returns:
            {mean_ms, std_ms, min_ms, max_ms}
        """
        desc = COCO_TASKS[task_key]["description"]
        times = []

        # Warm-up
        for _ in range(3):
            self.encode(desc)

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.encode(desc)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        return {
            "mean_ms": round(float(np.mean(times)), 2),
            "std_ms": round(float(np.std(times)), 2),
            "min_ms": round(float(np.min(times)), 2),
            "max_ms": round(float(np.max(times)), 2),
        }

    def measure_all_task_latencies(self, n_runs: int = 20) -> list[dict]:
        """Measure encoding latency for all 14 tasks."""
        results = []
        for task_key, task_info in COCO_TASKS.items():
            lat = self.measure_encoding_latency(task_key, n_runs)
            results.append({
                "task_key": task_key,
                "task_label": task_info["label"],
                **lat,
            })
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-modal fusion (task-guided detection filtering)
    # ─────────────────────────────────────────────────────────────────────────

    def filter_detections(self, detections: list[dict], task_key: str,
                          mode: str = "hard") -> dict:
        """
        Filter YOLO detections to keep only task-relevant objects.

        Two modes:
          "hard"  — binary filter: keep only detections whose class_id is
                    in the task's relevant_ids set. Simulates a simple lookup.
          "soft"  — confidence re-weighting: detections for task-relevant classes
                    get a 1.5× confidence boost, irrelevant ones get 0.3× penalty.
                    Returns all detections but re-ranked.

        Args:
            detections: list of dicts from YOLOInference.run()
            task_key:   one of the 14 COCO_TASKS keys
            mode:       "hard" or "soft"

        Returns:
            {
              "task_key": str,
              "task_label": str,
              "all_detections": list,       # original
              "filtered_detections": list,  # task-relevant only (hard) or re-ranked (soft)
              "n_before": int,
              "n_after": int,
              "suppression_rate_pct": float,
              "top_detection": dict | None,
            }
        """
        task = COCO_TASKS[task_key]
        relevant_ids = set(task["relevant_ids"])

        if mode == "hard":
            filtered = [d for d in detections if d["class_id"] in relevant_ids]

        elif mode == "soft":
            reweighted = []
            for d in detections:
                weight = 1.5 if d["class_id"] in relevant_ids else 0.3
                reweighted.append({
                    **d,
                    "task_score": round(d["confidence"] * weight, 4),
                })
            filtered = sorted(reweighted, key=lambda x: x["task_score"], reverse=True)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'hard' or 'soft'.")

        n_before = len(detections)
        n_after = len(filtered)
        suppression_rate = ((n_before - n_after) / n_before * 100
                            if n_before > 0 else 0.0)

        top = filtered[0] if filtered else None

        return {
            "task_key": task_key,
            "task_label": task["label"],
            "all_detections": detections,
            "filtered_detections": filtered,
            "n_before": n_before,
            "n_after": n_after,
            "suppression_rate_pct": round(suppression_rate, 1),
            "top_detection": top,
        }

    def evaluate_task_relevance(self, detections: list[dict],
                                 task_key: str) -> dict:
        """
        Compute task-relevance metrics for a set of YOLO detections:
          - Precision@k: fraction of top-k detections that are task-relevant
          - Recall: fraction of task-relevant classes detected at all

        Args:
            detections: sorted by confidence (highest first)
            task_key:   one of the 14 COCO_TASKS keys

        Returns:
            {precision_at_1, precision_at_3, precision_at_5, recall}
        """
        relevant_ids = set(COCO_TASKS[task_key]["relevant_ids"])

        def precision_at_k(k):
            top_k = detections[:k]
            if not top_k:
                return 0.0
            hits = sum(1 for d in top_k if d["class_id"] in relevant_ids)
            return round(hits / len(top_k), 3)

        detected_ids = {d["class_id"] for d in detections}
        detected_relevant = detected_ids & relevant_ids
        recall = (len(detected_relevant) / len(relevant_ids)
                  if relevant_ids else 0.0)

        return {
            "precision_at_1": precision_at_k(1),
            "precision_at_3": precision_at_k(3),
            "precision_at_5": precision_at_k(5),
            "recall": round(recall, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def print_filter_summary(self, filter_result: dict):
        print(f"\n  Task: {filter_result['task_label']} ({filter_result['task_key']})")
        print(f"  Detections before filter : {filter_result['n_before']}")
        print(f"  Detections after filter  : {filter_result['n_after']}")
        print(f"  Suppression rate         : {filter_result['suppression_rate_pct']:.1f}%")
        if filter_result["top_detection"]:
            top = filter_result["top_detection"]
            print(f"  Top detection            : {top['class_name']} "
                  f"(conf={top['confidence']:.3f})")
