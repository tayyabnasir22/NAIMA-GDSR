# NAIMA: Semantics Aware RGB Guided Depth Super-Resolution

Official implementation of **NAIMA** for **guided depth super-resolution (GDSR)**.

Paper: [arXiv:2604.04407](https://arxiv.org/abs/2604.04407)

## Citation

If you use this code or build upon the NAIMA-GDSR method in your research, please cite:

```bibtex
@article{nasir2026naimasemanticsawarergb,
  title   = {NAIMA: Semantics Aware RGB Guided Depth Super-Resolution},
  author  = {Nasir, Tayyab and Liu, Daochang and Mian, Ajmal},
  journal = {arXiv preprint arXiv:2604.04407},
  year    = {2026},
  url     = {https://doi.org/10.48550/arXiv.2604.04407}
}
```

## Requirements

Python 3.12 is recommended.

1. Create a new virtual environment:

```bash
python3.12 -m venv env
```

2. Activate the environment:

```bash
source env/bin/activate
```

3. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Training (`train.py`)

`train.py` is the **entry point for training**. It sets the global save root, selects the model and dataset, then delegates to `TrainingOrchestrator.Train()`.

**Command line**

```bash
python train.py <scale> <model_key>
```

| Argument      | Meaning |
|---------------|---------|
| `<scale>`     | Super-resolution scale factor (e.g. `4`, `8`, `16`). Passed to `TrainingOrchestrator.SCALE`. |
| `<model_key>` | Short key into an internal model map. Currently `v` selects `ModelType.NAIMA`. |

**What the script configures**

- `PathManager.BASE_PATH` — project-relative root for data and checkpoints (default `./`).
- `TrainingOrchestrator.SCALE` — upsampling factor.
- `TrainingOrchestrator.MODEL` — architecture to train (`NAIMA`).
- `TrainingOrchestrator.BENCHMARK` — training/validation data layout (`BenchmarkType.NYUV2` by default).
- `TrainingOrchestrator.REPEAT` / `TrainingOrchestrator.BATCH` — dataloader repeat factor and batch size (defaults in `train.py` override the class defaults in `TrainingOrchestrator` where applicable).

**Training behavior (orchestrator → trainer)**

`TrainingOrchestrator` dispatches to `NAIMATrainer`, which builds a `BaseNAIMATrainingPipeline` and runs `NAIMA_TrainingHelper`. Patch size is **420** for scale 4 and **448** for scales ≥ 8 (see `TrainingOrchestrator`). Checkpoints and logs are written under a directory produced by `PathManager.GetModelSavePath`, e.g.:

`./model_states_NAIMA_NYUV2_Patch_<patch>_Scale_<scale>/`

Place your benchmark data under a folder whose name matches the `BenchmarkType` value (e.g. `./NYUV2/` for NYU v2), consistent with `NAIMATrainer` paths.

**Data preparation:** scripts for preparing the benchmark datasets are available at [GDSR-Data-Preperation](https://github.com/tayyabnasir22/GDSR-Data-Preperation).

## Testing (`test.py`)

`test.py` is the **entry point for evaluation**. It calls static methods on `TestingOrchestrator` that construct a `NAIMA_Validator` with the appropriate `NAIMA_ValidationHelper` (RGB-guided) or `NAIMA_ValidationHelperBenchmark` (benchmark-style evaluation), then runs `TestModel` for each dataset and scale.

**Command line**

```bash
python test.py
```

**Default script behavior**

The provided `main()` runs a **fixed sequence** of evaluations: for each of several `BenchmarkType` values it runs 4×, 8×, and 16× (with separate code paths for standard vs. "benchmark" helpers where applicable). For day-to-day use, **comment out** the lines you do not need so only the desired dataset and scale run.

**Checkpoints**

Pretrained checkpoints are available [here](https://drive.google.com/drive/folders/1RNuYeMEkhs3dhOV4YPfbpIOc1dgp8TSd?usp=sharing).

## Repository structure

| Path | Role |
|------|------|
| `train.py` / `test.py` | CLI entry points for training and evaluation. |
| `Components/` | Network building blocks and the `NAIMA` module. |
| `Configurations/` | Dataclasses for training, validation, and data paths. |
| `DataProcessors/` | LR/HR and implicit downsampling logic for SR-style batches. |
| `ModelFactories/` | Maps `ModelType` to concrete architectures. |
| `Models/` | Enums and small utilities (`BenchmarkType`, `ModelType`, etc.). |
| `Pipelines/` | Training and validation pipeline bases (NAIMA-specific subclasses). |
| `TrainingOrchestrators/` / `TestingOrchestrators/` | High-level train/test dispatch. |
| `Trainers/` / `Validators/` | Trainer and validator implementations. |
| `TrainingHelpers/` / `ValidationHelpers/` | Epoch loops, metrics, and dataset-specific validation helpers. |
| `Utilities/` | Paths, logging, dataloaders, evaluation metrics, image I/O. |

## Acknowledgments

**Configurations, pipeline layout, and orchestration patterns** in this repository are adapted from the [INR-ASSR Empirical Analysis](https://github.com/tayyabnasir22/INR-ASSR-Emperical-Analysis) framework (training/testing orchestrators, dataclass-style configs, pipeline bases, validators, and related utilities). That project provides a unified interface for training and benchmarking INR-based arbitrary-scale super-resolution methods; this repo specializes it for NAIMA on GDSR benchmarks.

If you rely on that shared design or compare against INR-ASSR baselines, please also cite the INR-ASSR empirical study:

> **Implicit Neural Representation-Based Continuous Single Image Super-Resolution: An Empirical Benchmark**
> arXiv:2601.17723 — <https://arxiv.org/abs/2601.17723>

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).