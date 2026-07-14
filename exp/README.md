# Experiment Directory

This directory preserves the original experiment layout:

```text
exp/
├── cross_query_se/    # Main implementation and runnable scripts
├── DTR/               # Minimal DTR-compatible helpers for data loading and metrics
└── data/              # Local-only datasets and indexes; not committed
```

Run commands from this directory unless a script states otherwise:

```bash
cd exp
python cross_query_se/scripts/download_datasets.py
bash cross_query_se/scripts/verify_gpu_env.sh
```

The scripts infer `EXP_DIR` from their own location. You may override it manually:

```bash
EXP_DIR=/path/to/repo/exp bash cross_query_se/scripts/run_cross_query_se_sanity.sh
```

All local data, outputs, model caches, and generated result files are excluded from version control.
