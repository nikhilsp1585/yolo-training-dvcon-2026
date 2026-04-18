# Stage 2 Simulation — Task-Aware Sparse YOLO
**DVCon India 2026 | Submission #51 | Rhythm Patel & Nikhil Patel**

---

## File structure

```
stage2_sim/
├── requirements.txt        # Python dependencies
├── coco_tasks.py           # 14 task definitions (descriptions + COCO class IDs)
├── yolo_inference.py       # YOLOv5-nano inference + activation sparsity hooks
├── task_encoder.py         # DistilBERT encoder + cross-modal detection filter
├── run_simulation.py       # Main runner — orchestrates full pipeline
└── plot_results.py         # Generates all 7 report figures from results JSON
```

---

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running the simulation

### Full pipeline (50 images, ~10–20 min on CPU)
```bash
python run_simulation.py --n_images 50 --output_dir results/
```

### Quick test (10 images)
```bash
python run_simulation.py --n_images 10 --output_dir results/
```

### GPU acceleration
```bash
python run_simulation.py --n_images 100 --device cuda --output_dir results/
```

### Soft filtering mode (confidence re-weighting instead of binary filter)
```bash
python run_simulation.py --n_images 50 --filter_mode soft --output_dir results/
```

---

## Generating report figures

After `run_simulation.py` completes:

```bash
python plot_results.py --results results/simulation_results.json
```

This produces 7 PNG figures in `results/`:

| Figure | Content |
|--------|---------|
| `fig1_per_layer_sparsity.png`   | Top-20 Conv/ReLU layers by sparsity % |
| `fig2_sparsity_mac_summary.png` | Overall sparsity + MAC reduction vs proposed claim range |
| `fig3_encoding_latency.png`     | DistilBERT encoding time per task (ms) |
| `fig4_precision_per_task.png`   | Precision@1 and Precision@3 per task |
| `fig5_suppression_rate.png`     | Task-irrelevant object suppression rate per task |
| `fig6_latency_comparison.png`   | Sequential vs pipelined latency + speedup factor |
| `fig7_summary_dashboard.png`    | All key metrics in one 2×2 dashboard |

---

## Expected results (based on Stage 1 report claims)

| Metric | Expected |
|--------|----------|
| Activation sparsity (YOLOv5-nano) | 30–50% |
| MAC reduction via zero-skipping   | 30–50% |
| Detection relevance improvement   | ~18–22% |
| Pipelined speedup (5-image batch) | ~4–5×  |
| Inference latency (pipelined)     | ~63.8 ms/image |
| DistilBERT encoding latency       | ~20–60 ms (CPU) |

---

## Modules in detail

### `yolo_inference.py`
- Loads `yolov5n.pt` via `ultralytics`
- Registers `torch.nn.Module` forward hooks on every `Conv2d / ReLU / SiLU`
- Counts zero-valued activations post-ReLU to compute sparsity %
- Estimates dense vs effective (zero-skipped) MAC count

### `task_encoder.py`
- Loads `distilbert-base-uncased` from HuggingFace
- Encodes task descriptions → 768-d L2-normalised CLS embeddings
- Hard filter: keeps only detections whose class_id ∈ task's relevant_ids
- Soft filter: re-weights confidences (1.5× for relevant, 0.3× for irrelevant)
- Computes Precision@1/3/5 and Recall for task-relevance evaluation

### `run_simulation.py`
- Downloads COCO val2017 sample images automatically
- Runs all steps and saves `results/simulation_results.json`
- Prints a concise summary table at the end

### `plot_results.py`
- Reads the JSON and produces 7 publication-ready matplotlib figures
- All figures use consistent colour scheme matching the Stage 1 architecture diagrams

---

## Troubleshooting

**`ultralytics` downloads model on first run** — YOLOv5n (~4 MB) is downloaded
automatically to `~/.ultralytics/` on the first call. Requires internet.

**DistilBERT model download** — ~260 MB, cached to `~/.cache/huggingface/` on first run.

**Slow on CPU** — encoding latency on CPU is 30–100 ms per task. Use `--device cuda`
if a GPU is available for ~10× speedup. The sparsity measurement is the bottleneck
on CPU; reduce `--n_images` for a quick test.

**Hook compatibility** — hooks are registered on the ultralytics internal PyTorch model
(`model.model.model`). If you upgrade `ultralytics`, the internal structure may change;
check `yolo_inference.py` line ~55 if hooks show 0 layers registered.
