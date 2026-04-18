"""
yolo_inference.py
─────────────────
YOLOv5-nano inference wrapper with:
  1. Standard object detection on COCO images
  2. Activation capture hooks on every Conv/ReLU layer
  3. Per-layer sparsity measurement (fraction of zero activations)
  4. MAC operation counting (dense vs effective after zero-skipping)

Usage:
    from yolo_inference import YOLOInference
    model = YOLOInference()
    results = model.run("path/to/image.jpg")
    sparsity = model.get_sparsity_report()
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


class ActivationCaptureHook:
    """Registers a forward hook on a layer to capture output activations."""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.activation: Optional[torch.Tensor] = None

    def hook_fn(self, module, input, output):
        # Detach to avoid holding the computation graph
        self.activation = output.detach().cpu()

    def clear(self):
        self.activation = None


class YOLOInference:
    """
    Wraps YOLOv5-nano (via ultralytics) for:
      - Task-agnostic detection (baseline)
      - Activation sparsity measurement across all Conv layers
      - MAC counting: dense vs sparse (zero-skipped)
    """

    def __init__(self, model_name: str = "yolov5n.pt", conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = "auto"):
        print(f"[YOLOInference] Loading {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[YOLOInference] Running on: {self.device}")

        # Hook infrastructure
        self._hooks: list = []
        self._capture_hooks: list[ActivationCaptureHook] = []
        self._layer_names: list[str] = []
        self._hooks_registered = False

    # ─────────────────────────────────────────────────────────────────────────
    # Hook registration
    # ─────────────────────────────────────────────────────────────────────────

    def register_activation_hooks(self):
        """
        Attach forward hooks to every Conv2d and BatchNorm2d layer in the
        YOLO backbone + neck. This lets us capture activations post-ReLU.
        """
        if self._hooks_registered:
            return

        # Access the underlying PyTorch model
        pt_model = self.model.model.model  # ultralytics internal structure

        for name, module in pt_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.SiLU)):
                capture = ActivationCaptureHook(name)
                handle = module.register_forward_hook(capture.hook_fn)
                self._hooks.append(handle)
                self._capture_hooks.append(capture)
                self._layer_names.append(name)

        self._hooks_registered = True
        print(f"[YOLOInference] Registered hooks on {len(self._hooks)} layers.")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._capture_hooks.clear()
        self._layer_names.clear()
        self._hooks_registered = False

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, image_source, verbose: bool = False):
        """
        Run YOLOv5-nano inference.

        Args:
            image_source: path string, PIL image, or numpy array
            verbose: print raw YOLO output

        Returns:
            list of dicts: [{class_id, class_name, confidence, bbox}, ...]
        """
        results = self.model.predict(
            source=image_source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose,
            device=self.device,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": float(box.conf.item()),
                    "bbox": box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
                })

        return detections

    def run_batch(self, image_paths: list, verbose: bool = False) -> list[list[dict]]:
        """Run inference on a list of images and return per-image detections."""
        all_detections = []
        for path in image_paths:
            dets = self.run(path, verbose=verbose)
            all_detections.append(dets)
        return all_detections

    # ─────────────────────────────────────────────────────────────────────────
    # Sparsity measurement
    # ─────────────────────────────────────────────────────────────────────────

    def measure_sparsity(self, image_source) -> dict:
        """
        Run one forward pass with hooks active and compute per-layer sparsity.

        Sparsity = fraction of activations that are exactly zero (post-ReLU/SiLU).

        Returns:
            {
              "per_layer": [{name, total_elements, zero_elements, sparsity_pct}, ...],
              "overall_sparsity_pct": float,
              "dense_macs_estimate": int,
              "effective_macs_estimate": int,
              "mac_reduction_pct": float,
            }
        """
        if not self._hooks_registered:
            self.register_activation_hooks()

        # Clear previous captures
        for ch in self._capture_hooks:
            ch.clear()

        # Forward pass (activations captured via hooks)
        self.run(image_source, verbose=False)

        per_layer = []
        total_elements = 0
        total_zeros = 0

        for ch in self._capture_hooks:
            if ch.activation is None:
                continue

            act = ch.activation.float()
            n_elements = act.numel()
            n_zeros = int((act == 0).sum().item())
            sparsity = n_zeros / n_elements if n_elements > 0 else 0.0

            per_layer.append({
                "layer_name": ch.layer_name,
                "total_elements": n_elements,
                "zero_elements": n_zeros,
                "sparsity_pct": round(sparsity * 100, 2),
            })

            total_elements += n_elements
            total_zeros += n_zeros

        overall_sparsity = total_zeros / total_elements if total_elements > 0 else 0.0

        # MAC estimation
        # Dense MACs ≈ total_elements (each output element = 1 MAC in simplified model)
        dense_macs = total_elements
        # Effective MACs after zero-skipping = non-zero activations only
        effective_macs = total_elements - total_zeros
        mac_reduction = (1 - effective_macs / dense_macs) * 100 if dense_macs > 0 else 0.0

        return {
            "per_layer": per_layer,
            "overall_sparsity_pct": round(overall_sparsity * 100, 2),
            "dense_macs_estimate": dense_macs,
            "effective_macs_estimate": effective_macs,
            "mac_reduction_pct": round(mac_reduction, 2),
        }

    def measure_sparsity_batch(self, image_paths: list) -> dict:
        """
        Average sparsity across multiple images for a stable estimate.

        Returns the same structure as measure_sparsity() but averaged.
        """
        all_results = [self.measure_sparsity(p) for p in image_paths]

        # Average overall sparsity
        avg_sparsity = np.mean([r["overall_sparsity_pct"] for r in all_results])
        avg_mac_reduction = np.mean([r["mac_reduction_pct"] for r in all_results])
        avg_dense = int(np.mean([r["dense_macs_estimate"] for r in all_results]))
        avg_effective = int(np.mean([r["effective_macs_estimate"] for r in all_results]))

        # Average per-layer (align by layer name)
        layer_names = [l["layer_name"] for l in all_results[0]["per_layer"]]
        per_layer_avg = []
        for i, name in enumerate(layer_names):
            vals = [r["per_layer"][i]["sparsity_pct"] for r in all_results
                    if i < len(r["per_layer"])]
            per_layer_avg.append({
                "layer_name": name,
                "sparsity_pct": round(float(np.mean(vals)), 2),
            })

        return {
            "per_layer": per_layer_avg,
            "overall_sparsity_pct": round(float(avg_sparsity), 2),
            "dense_macs_estimate": avg_dense,
            "effective_macs_estimate": avg_effective,
            "mac_reduction_pct": round(float(avg_mac_reduction), 2),
            "num_images_averaged": len(image_paths),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def print_sparsity_summary(self, sparsity_report: dict):
        print("\n" + "="*60)
        print("  YOLO ACTIVATION SPARSITY REPORT")
        print("="*60)
        print(f"  Overall sparsity      : {sparsity_report['overall_sparsity_pct']:.1f}%")
        print(f"  Dense MACs (est.)     : {sparsity_report['dense_macs_estimate']:,}")
        print(f"  Effective MACs (est.) : {sparsity_report['effective_macs_estimate']:,}")
        print(f"  MAC reduction         : {sparsity_report['mac_reduction_pct']:.1f}%")
        if "num_images_averaged" in sparsity_report:
            print(f"  Images averaged       : {sparsity_report['num_images_averaged']}")
        print("="*60)
        print(f"\n  Top-10 sparsest layers:")
        sorted_layers = sorted(sparsity_report["per_layer"],
                               key=lambda x: x["sparsity_pct"], reverse=True)
        for l in sorted_layers[:10]:
            bar = "█" * int(l["sparsity_pct"] / 5)
            print(f"    {l['layer_name']:<40} {l['sparsity_pct']:5.1f}%  {bar}")
        print()
