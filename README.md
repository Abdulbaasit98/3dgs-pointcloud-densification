# Neighbor-Interpolated Point Cloud Densification for 3D Gaussian Splatting

> **Thesis project** — ABDUGANIEV ABDULVOSIT, Seoul National University of Science and Technology, 2026  
> Supervisor: [Supervisor Name]

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] initializes one Gaussian per point in a sparse
Structure-from-Motion (SfM) point cloud produced by COLMAP. Under-sampled regions (textureless
surfaces, reflective areas) start with too few Gaussians and rely entirely on Adaptive Density
Control (ADC) to fill gaps during training.

This project proposes a simple, **zero-cost-at-training-time** preprocessing step:
**gap-filling point cloud densification** via k-nearest-neighbor midpoint interpolation,
applied to the SfM point cloud before Gaussian initialization.

```
Original SfM points  →  densify_init_pointcloud.py  →  Denser .ply  →  3DGS training (unmodified)
     136,029 pts                                         197,869 pts
```

No changes to the 3DGS training code or CUDA rasterizer are required.

---

## Key Results (Tanks & Temples "truck" scene, iteration 7000)

| Iteration | PSNR Base | PSNR Dense | ΔPSNR | SSIM Base | SSIM Dense | ΔSSIM | LPIPS Base | LPIPS Dense | ΔLPIPS |
|-----------|-----------|------------|-------|-----------|------------|-------|------------|-------------|--------|
| 1,000 | 20.978 | 21.370 | **+0.391** | 0.7315 | 0.7538 | **+0.022** | 0.3542 | 0.3264 | **−0.028** |
| 3,000 | 22.728 | 22.966 | **+0.238** | 0.8074 | 0.8183 | **+0.011** | 0.2524 | 0.2368 | **−0.016** |
| 7,000 | 23.988 | 24.154 | **+0.165** | 0.8528 | 0.8595 | **+0.007** | 0.1950 | 0.1846 | **−0.010** |

- Densified initialization wins on all three metrics at every checkpoint.
- The advantage is **largest early in training** (ΔPSNR = +0.39 at iter 1000) and narrows as both
  models converge — consistent with the hypothesis that denser init provides an early-training
  head start that ADC gradually compensates for.
- Per-view robustness: densified better in 17/32 test views (Wilcoxon p = 0.49 — effect present
  in aggregate metrics but not statistically significant at the per-view level with n=32).
- Final Gaussian count: baseline 1,681,300 vs. densified 1,733,031 — **no compactness benefit**;
  ADC growth is largely independent of initialization density.

---

## Repository Structure

```
gs_densify_init/
├── scripts/
│   ├── densify_init_pointcloud.py   # core contribution: gap-filling densification
│   └── compare_results.py           # parse & compare two results.json files
├── notebooks/
│   └── reproduce_experiment.ipynb   # full Colab-ready notebook, run top-to-bottom
├── results/
│   └── truck_results.json           # PSNR/SSIM/LPIPS for both runs (all checkpoints)
├── figures/
│   ├── convergence_curve.png        # [TODO: add after generating]
│   └── qualitative_comparison.png   # [TODO: add from Colab output]
├── docs/
│   └── thesis_draft.pdf             # thesis write-up (draft)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quickstart

### 1. Install dependencies (CPU only, no GPU needed for preprocessing)

```bash
git clone https://github.com/Abdulbaasit98/gs_densify_init.git
cd gs_densify_init
pip install -r requirements.txt
```

### 2. Clone the base 3DGS repo

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

### 3. Densify your point cloud

```bash
# Input: points3D.bin or points3D.ply from COLMAP sparse reconstruction
python scripts/densify_init_pointcloud.py \
    --input path/to/sparse/0/points3D.bin \
    --output path/to/sparse_densified/0/points3D.ply \
    --k 6 \
    --factor 3.0 \
    --max_new_per_point 1
```

This prints the original and densified point counts and saves the result as a `.ply` file
compatible with 3DGS's `fetchPly()` loader.

### 4. Train (unmodified 3DGS)

```bash
# Baseline
python gaussian-splatting/train.py \
    -s data/scene_baseline \
    -m output/scene_baseline \
    --eval \
    --iterations 7000 \
    --test_iterations 1000 3000 7000 \
    --save_iterations 1000 3000 7000

# Densified init (same command, different -s)
python gaussian-splatting/train.py \
    -s data/scene_densified \
    -m output/scene_densified \
    --eval \
    --iterations 7000 \
    --test_iterations 1000 3000 7000 \
    --save_iterations 1000 3000 7000
```

### 5. Evaluate and compare

```bash
python gaussian-splatting/render.py -m output/scene_baseline --iteration 1000
python gaussian-splatting/render.py -m output/scene_baseline --iteration 3000
python gaussian-splatting/render.py -m output/scene_baseline --iteration 7000
python gaussian-splatting/metrics.py -m output/scene_baseline

python gaussian-splatting/render.py -m output/scene_densified --iteration 1000
python gaussian-splatting/render.py -m output/scene_densified --iteration 3000
python gaussian-splatting/render.py -m output/scene_densified --iteration 7000
python gaussian-splatting/metrics.py -m output/scene_densified

python scripts/compare_results.py \
    --baseline output/scene_baseline/results.json \
    --modified output/scene_densified/results.json
```

Or use the Colab notebook: `notebooks/reproduce_experiment.ipynb` — runs the full pipeline
top-to-bottom in a free-tier T4 session.

---

## Method

```
COLMAP sparse .bin
        │
        ▼
  load_points3D()          reads xyz + rgb from .bin or .ply
        │
        ▼
  cKDTree(xyz)             k-NN on CPU
        │
        ▼
  median neighbor dist     scene-scale reference
        │
        ▼
  gap detection            pair distance > factor × median → insert midpoint
        │
        ▼
  save_ply()               writes x,y,z,nx,ny,nz,r,g,b  (3DGS fetchPly schema)
        │
        ▼
  train.py (unmodified)    fetches densified .ply as Gaussian initialization
```

**Key hyperparameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--k` | 6 | Neighbors considered per point |
| `--factor` | 3.0 | Gap threshold multiplier (higher = fewer, more conservative insertions) |
| `--max_new_per_point` | 1 | Cap on new points per original point |

---

## Reproduce on Google Colab (free tier)

Open `notebooks/reproduce_experiment.ipynb` directly in Colab. The notebook handles:
- Repo clone and CUDA extension build
- Dataset download (Tanks & Temples "truck")
- Densification preprocessing
- Training both conditions (capped at 7000 iterations for free-tier feasibility)
- Evaluation and comparison table

**Important:** Mount Google Drive before training to preserve checkpoints across session resets.

---

## Limitations

- Single scene evaluated (Tanks & Temples "truck"), single seed.
- No hyperparameter sweep; a full ablation over `--factor` and `--k` remains future work.
- Training capped at 7000 iterations due to Colab free-tier constraints (paper uses 30000).
- Per-view improvement not statistically significant at n=32 (Wilcoxon p=0.49); aggregate
  metric improvement should be interpreted as a modest, directional effect.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{yourname2026densifyinit,
  author    = {ABDUGANIEV ABDULVOSIT},
  title     = {Neighbor-Interpolated Point Cloud Densification for 3D Gaussian Splatting Initialization},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/Abdulbaasit98/gs_densify_init}
}
```

And the base 3DGS paper:

```bibtex
@article{kerbl3Dgaussians,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  volume    = {42},
  number    = {4},
  year      = {2023}
}
```

---

## Acknowledgements

Built on top of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
by Kerbl et al. Dataset: [Tanks and Temples](https://www.tanksandtemples.org/).
