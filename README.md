# Numerical Fragility in Transformers

Official code release for the AISTATS 2026 main-conference paper:

**Numerical Fragility in Transformers: A Layer-wise Theory for Risk Estimation and Selective Stabilization**

This repository contains the **official experimental code** for the paper. It focuses on the shared utilities, experiment runners, and plotting scripts needed to reproduce the paper's main empirical results.

This public code release does **not** include:

- the camera-ready LaTeX source
- precomputed experiment outputs
- paper-only figure copies

Users are expected to run the experiments locally or on a cluster and generate outputs from scratch.

## Included Experiments

The final paper is built around four experiments:

- **E1**: controlled local mechanism validation
- **E2**: end-to-end predictor validation on GPT-2
- **E3**: attribution and localization fidelity
- **E5**: budgeted mitigation utility under BGSS

The repository also includes shared code used across these experiments:

- `common/`: shared utilities and manual GPT-2 forward helpers
- `run_all_experiments.sh`: orchestration script
- `render_publication_plots.py`: publication-plot renderer

## Public Repository Layout

```text
.
|-- README.md
|-- common/
|-- e1_controlled/
|-- e2_predictor/
|-- e3_attribution/
|-- e5_bgss/
|-- render_publication_plots.py
`-- run_all_experiments.sh
```

## Environment

The code expects:

- Python `3.10+`
- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- `matplotlib`
- a Bash-compatible shell for orchestration

A minimal setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets matplotlib
```

For `E2`, `E3`, and `E5`, GPU execution is strongly recommended.

## Quick Start

Run the main paper experiments from the repository root:

```bash
bash run_all_experiments.sh e1 e2 e3 e5
```

This creates per-experiment run directories and summary outputs under the corresponding experiment folders.

To regenerate publication-ready plots from completed runs:

```bash
python render_publication_plots.py --experiments e1 e2 e3 e5
```

## Experiment Map

### E1: Controlled Local Validation

Location:

- [e1_controlled](e1_controlled)

Purpose:

- validate local mechanism predictions in isolation
- check attention proxy scaling
- verify LayerNorm monotonicity in `epsilon`
- verify residual transport behavior with depth

### E2: GPT-2 End-to-End Predictor Validation

Location:

- [e2_predictor](e2_predictor)

Default setup:

- model: `gpt2`
- dataset: `wikitext-103-v1` validation
- target precisions: `bf16`, `fp16`
- sequence lengths: `128`, `512`, `1024`
- seeds: `0`, `1`, `2`

Purpose:

- test whether the transport-aware combined predictor tracks FP32-reference mismatch on real GPT-2 windows

### E3: Attribution and Localization Fidelity

Location:

- [e3_attribution](e3_attribution)

Purpose:

- compare the practical layer-wise proxy with exact-ish reference-patch attribution on the highest-mismatch windows selected from E2

### E5: BGSS Budgeted Mitigation

Location:

- [e5_bgss](e5_bgss)

Default setup:

- FP32 master training
- FP16 shadow monitoring
- GPT-2 on `wikitext-2-raw-v1` train
- sequence length `256`
- seeds `0`, `1`, `2`
- policies: `none`, `static_global`, `random_same_budget`, `bgss`

Purpose:

- test whether the theory-guided BGSS controller improves budget-matched mitigation relative to standard baselines

## Reproducibility Notes

- Each experiment has its own `configs/default.json`.
- `run_all_experiments.sh` accepts per-experiment config overrides via:
  - `E1_CONFIG`
  - `E2_CONFIG`
  - `E3_CONFIG`
  - `E4_CONFIG`
  - `E5_CONFIG`

Example:

```bash
E2_CONFIG=/path/to/custom_e2.json bash run_all_experiments.sh e2
```

- Experiment outputs are generated locally and are not part of the public repository.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{baek2026numericalfragility,
  title     = {Numerical Fragility in Transformers: A Layer-wise Theory for Risk Estimation and Selective Stabilization},
  author    = {Baek, Jinwoo},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics},
  year      = {2026}
}
```

## License

Please add a license file before making the repository public. A simple and common choice for research code is **MIT**, while **Apache-2.0** is a good option if you want an explicit patent grant.

The license file should live at the repository root and be named exactly:

```text
LICENSE
```

If you want, I can also draft the full `LICENSE` file for either:

- `MIT`
- `Apache-2.0`

## Contact

Questions about the paper or code release can be directed to the author through the paper contact information.
