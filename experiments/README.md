# Experiments workflow

Define experiments as data, run them as SLURM array jobs, and move files with
rsync instead of git. Three layers:

| File | Role |
|---|---|
| `experiments/sweeps.py` | **The only file you edit** — one `Sweep(...)` per experiment (grid + SLURM resources). |
| `experiments/run.py` | Runs one config from a sweep, selected by array-task index. |
| `experiments/submit.py` | Builds & submits the `sbatch` array (computes the array size from the grid). |
| `jobs/array.sh` | Generic array script (never edited); logs provenance, dispatches to `run.py`. |
| `scripts/deploy.sh` / `scripts/fetch.sh` | rsync code up / results down. |

## Add an experiment

Add one entry to `SWEEPS` in `experiments/sweeps.py`. Grid keys are keyword
arguments of `run_mto_experiment` — lists are swept, scalars are fixed:

```python
"axiom2_bpic17dr": Sweep(
    grid=dict(
        log_name="BPIC_17_DR", mto_technique="equal_weighting",
        seed=[103, 104, 105],
        lambda_ltn_outcome=[0.0, 0.1, 0.25, 0.5],
        detach_mode_outcome=["none", "act", "outcome"],
        subset_fraction=0.5, val_subset_fraction=0.5,
        num_epochs=40, patience=8,
        batch_size=256, validate_every=2,      # perf knobs, tune from a profile run
    ),
    partition="tue.gpu1.q", gres="gpu:1g.6gb:1", time="08:00:00", concurrency=6,
),
```

Check it locally before deploying:

```bash
python -m experiments.run axiom2_bpic17dr      # lists configs; [exists] marks done ones
```

A `lambda*=0.0` baseline swept over `detach_*` values collapses to a single
baseline automatically (same folder name → deduped).

## Run it on the cluster

```bash
# 1. from the laptop: upload code (stamps the commit)
scripts/deploy.sh you@login.hpc.tue.nl

# 2. on the server (first time / after dependency changes)
cd /scratch-shared/$USER/thesis && uv sync

# 3. on the server: submit the array (array size computed from the grid)
python -m experiments.submit axiom2_bpic17dr            # or --dry-run to preview
squeue -u $USER                                         # monitor

# 4. back on the laptop: pull results + checkpoints for one log
scripts/fetch.sh you@login.hpc.tue.nl BPIC_17_DR
```

Re-running `submit` resumes: configs whose result folder already exists are
skipped. To force a rerun, delete that folder on the server first.

## One-time data upload

`deploy.sh` excludes data (`.deployignore`), so upload the preprocessed tensors
once, directly:

```bash
rsync -avz BPIC_17_DR/ you@login.hpc.tue.nl:/scratch-shared/$USER/thesis/BPIC_17_DR/ \
  --include='*/' --include='*.pt' --include='*.pkl' --exclude='*'
```

(Or generate them on the server by uploading the raw `*.csv` and running the
`create_*` scripts there.)

## Profiling before a real sweep

`submit profile` runs a 2-epoch job on the L4 (which reports GPU utilisation).
After it finishes, `fetch` the `logs/` and read the mean GPU util + epoch time,
then set `batch_size` / `validate_every` / `val_subset_fraction` in the real
sweep accordingly. Low util → raise `batch_size` and/or validate less often.

## Perf knobs (in the grid)

- `batch_size` — raise it for the tiny model until GPU util is healthy.
- `validate_every` — run the (expensive, autoregressive) validation every N
  epochs instead of every epoch. Checkpoints are still saved every epoch;
  early-stopping `patience` then counts *validation events*, so `patience`
  effectively covers `validate_every * patience` epochs.
- `val_subset_fraction` — validate on a case-level subset of the val set.

## rsync on Windows

`deploy.sh` / `fetch.sh` need `rsync` + `ssh`. Easiest: run them from **WSL**
(`sudo apt install rsync`) or **Git Bash** with an rsync package. Set up SSH
key auth to the login node first so transfers don't prompt for a password.
