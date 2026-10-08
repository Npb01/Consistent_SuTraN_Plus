# Experiments workflow

Define experiments as data, run them as SLURM array jobs, and move files with
rsync instead of git. Layers:

| File | Role |
|---|---|
| `experiments/sweeps.py` | **The only file you edit** — one `Sweep(...)` per experiment (grid + SLURM resources). |
| `experiments/run.py` | Runs one config from a sweep, selected by array-task index. |
| `experiments/submit.py` | Builds & submits the `sbatch` array (computes the array size from the grid). |
| `jobs/array.sh` | Generic, cluster-agnostic array script (never edited); logs provenance, dispatches to `run.py`. |
| `scripts/deploy.sh` / `scripts/fetch.sh` | rsync code up / results down. |

## Cluster configuration (one-time)

All site-specific settings live in environment variables, never in the code.
Copy the template, fill in your values, and `source` it in any shell that runs
`deploy`/`fetch` (locally) or `submit` (on the server):

```bash
cp experiments/cluster.example.env experiments/cluster.env   # cluster.env is git-ignored
# edit experiments/cluster.env, then:
source experiments/cluster.env
```

- `REMOTE_DIR` — where the code + data live on the cluster (rsync target for
  `deploy`/`fetch`, and the working directory the SLURM jobs run in). A relative
  path is under the remote home; use an absolute path for a scratch/project
  filesystem.
- `SLURM_PARTITION`, `SLURM_GRES`, `SLURM_ACCOUNT` — when set, override the
  per-sweep resource defaults declared in `sweeps.py`, so a new cluster needs no
  edits there. `SLURM_ACCOUNT` is only passed when non-empty.

The SSH host is the **first argument** to `deploy.sh`/`fetch.sh`. For
convenience define an alias in `~/.ssh/config` (not committed) and pass that:

```
Host mycluster
    HostName login.my-hpc.example
    User myusername
    IdentityFile ~/.ssh/id_ed25519
```

## Where to run each command

Every command block below is tagged with one of two contexts:

- **[local]** — a shell on your workstation: file transfers (`deploy.sh`,
  `fetch.sh`, `rsync`) and local Python (`collect_results`, `analyze`, listing
  configs). Transfers need `rsync` + `ssh` installed locally. On Windows, run
  the transfer/`ssh` lines from WSL (see "Windows / transfers" at the end).
- **[server]** — a shell on the cluster login node: job control (`uv sync`,
  `submit`, `squeue`, `run`).

## Add an experiment

Add one entry to `SWEEPS` in `experiments/sweeps.py`. Grid keys are keyword
arguments of `run_mto_experiment` — lists are swept, scalars are fixed:

```python
"my_sweep": Sweep(
    grid=dict(
        log_name="BPIC_17_DR", mto_technique="equal_weighting",
        seed=[103, 104, 105],
        lambda_ltn_outcome=[0.0, 0.1, 0.25, 0.5],
        detach_mode_outcome=["none", "act", "outcome"],
        subset_fraction=0.5, val_subset_fraction=0.5,
        num_epochs=40, patience=8,
        batch_size=256, validate_every=2,      # perf knobs, tune from a profile run
    ),
    # Resource defaults; SLURM_PARTITION / SLURM_GRES env vars override these.
    partition="gpu", gres="gpu:1", time="08:00:00", concurrency=6,
),
```

Check it locally before deploying:

```bash
# [local]
uv run python -m experiments.run my_sweep   # lists configs; [exists] marks done ones
```

A `lambda*=0.0` baseline swept over `detach_*` values collapses to a single
baseline automatically (same folder name → deduped).

## Run it on the cluster

```bash
# [local] 1. upload code (stamps the commit). Set your checkout as the cwd:
cd /path/to/your/checkout
source experiments/cluster.env          # sets REMOTE_DIR etc.
bash scripts/deploy.sh mycluster

# [local] 2. hop to the server
ssh mycluster
```
```bash
# [server] 3. first time / after dependency changes
source experiments/cluster.env
cd "$REMOTE_DIR" && uv sync

# [server] 4. submit the array (array size computed from the grid)
uv run python -m experiments.submit my_sweep --dry-run   # preview the sbatch command
uv run python -m experiments.submit my_sweep
squeue -u "$USER"                                        # monitor
```
```bash
# [local] 5. back on the workstation (exit the server): pull results for one log
cd /path/to/your/checkout
bash scripts/fetch.sh mycluster BPIC_17_DR
```

Re-running `submit` resumes: configs whose result folder already exists are
skipped. To force a rerun, delete that folder on the server first.

## Collect & analyse results

After `fetch`, gather every run's artifacts into tidy CSVs with
`collect_results` (sweep-driven; it resolves each run dir, so no name parsing).
See [`SAVED_OUTPUTS.md`](SAVED_OUTPUTS.md) for what each column means.

```bash
# [local] per-run SUMMARY: one row per run (config + best-epoch metrics + test
# metrics + compute cost + lambda contribution share). Several sweeps at once.
uv run python -m experiments.collect_results axiom2_impl_uw axiom2_impl_pilot --out collected_results.csv

# [local] ALSO emit the per-epoch CURVES (one row per run x epoch: training losses,
# validation metrics, compute cost) for training graphs:
uv run python -m experiments.collect_results axiom2_impl_uw --history history.csv
```

Missing runs appear as `found=False` rows, so you can see at a glance what still
needs to run. In a notebook, import the functions instead of the CLI:

```python
from experiments.collect_results import collect_sweeps, collect_history
summary = collect_sweeps(["axiom2_impl_uw"])          # per-run
history = collect_history(["axiom2_impl_uw"])         # per-epoch curves
```

For a quick per-impl comparison table from the summary CSV (means over seeds):

```bash
# [local]
uv run python -m experiments.analyze collected_results.csv --axiom 2
```

`analyze.py` is a starting point — extend it with best-of-lambda selection,
significance tests, and plots.

For the per-instance **inconsistency ↔ error** analysis (is per-instance
inconsistency a reliability signal?), use `consistency_per_instance`, which reads
each run's `TEST_SET_RESULTS/consistency_per_instance.pt` (see
[`SAVED_OUTPUTS.md`](SAVED_OUTPUTS.md)) and emits a per-run reliability summary
(ax1 |gap|-vs-|error| correlation + decile lift; ax2 head accuracy by
agree/disagree) plus optional per-instance long tables:

```bash
# [local]
uv run python -m experiments.consistency_per_instance axiom2_impl_uw \
    --summary reliability_summary.csv --out-ax2 ax2_per_instance.csv
```

Only runs trained after per-instance logging was added have that file; others
are skipped.

## Data: persistent master, working copy on scratch

Many clusters purge their fast scratch filesystem on a timer (by file
timestamp), and `rsync -a` *preserves* mtimes — so freshly uploaded tensors can
be swept almost immediately if they were created long ago. The robust pattern:
keep a **master copy on a permanent filesystem** (home or a project allocation)
and stage a **working copy into `$REMOTE_DIR` with `cp`**, which stamps a fresh
mtime. `deploy.sh` never touches data (it is in `.deployignore`).

```bash
# [local] once: upload the tensors to a PERSISTENT location on the cluster
#         (exclude the result dirs that share BPIC_17_DR/; -z omitted, it can stall)
rsync -av --exclude='SUTRAN_DA_results_*/' \
  /path/to/your/checkout/BPIC_17_DR/ \
  mycluster:~/thesis_data/BPIC_17_DR/
```
```bash
# [server] stage into the working dir where the code expects it (fresh mtime)
source experiments/cluster.env
mkdir -p "$REMOTE_DIR/BPIC_17_DR"
cp ~/thesis_data/BPIC_17_DR/* "$REMOTE_DIR/BPIC_17_DR/"
ls "$REMOTE_DIR"/BPIC_17_DR/*.pkl       # sanity: the cardinality/mapping dicts are there
```

(No filesystem here is backed up, so keep the real source elsewhere too.
Alternatively, generate the tensors on the server from the raw `*.csv` with the
`create_*` scripts.)

## Recovery after a scratch purge

A purge removes the code, the staged tensors, the venv, **and any results you
have not fetched** — so `fetch` results before any gap. To resume (the permanent
master survives, so there is no network re-upload):

```bash
# [local] 1. re-deploy code
cd /path/to/your/checkout
source experiments/cluster.env
bash scripts/deploy.sh mycluster
```
```bash
# [server] 2. re-stage data from the master (local cp = fresh mtime), then
#            rebuild the venv and sanity-check torch
source experiments/cluster.env
mkdir -p "$REMOTE_DIR/BPIC_17_DR"
cp ~/thesis_data/BPIC_17_DR/* "$REMOTE_DIR/BPIC_17_DR/"
cd "$REMOTE_DIR" && rm -rf .venv && uv sync
uv run python -c "import torch; print(torch.__file__); print(torch.cuda.is_available())"
```

If `torch.__file__` prints `None`, the venv did not materialise (see the uv note
in the TU/e example below) — `rm -rf .venv && uv sync` again.

## Profiling batch size before a real sweep

GPU **utilisation** is the textbook signal, but a shared MIG slice often reports
it as `N/A`, so profile by **epoch time** instead (`train_seconds` +
`peak_gpu_mem_mb` per epoch are logged in `compute_cost.csv`). Per epoch the
sample count is fixed, so a bigger batch means fewer steps and less per-step
overhead — if `train_seconds` keeps **dropping** as batch size rises you were
GPU-starved (raise it); where it **flattens** is the sweet spot; keep
`peak_gpu_mem_mb` well under the device's memory.

```bash
# [server]
uv run python -m experiments.submit batch_size_probe
```
```bash
# [local] fetch, then:
uv run python -m experiments.collect_results batch_size_probe --out bs.csv
# compare the batch_size / train_seconds_mean / peak_gpu_mem_mb_max columns
```

Then **fix the chosen `batch_size` across every real sweep** — the compute-cost
comparison across impls needs it constant.

## Perf knobs (in the grid)

- `batch_size` — raise it for the tiny model until `train_seconds` stops
  dropping (see "Profiling batch size"); keep it constant across the comparison.
- `validate_every` — run the (expensive, autoregressive) validation every N
  epochs instead of every epoch. Checkpoints are still saved every epoch;
  early-stopping `patience` then counts *validation events*, so `patience`
  effectively covers `validate_every * patience` epochs.
- `val_subset_fraction` — validate on a case-level subset of the val set.

## Example: the TU/e HPC setup used for this thesis

One concrete instantiation of the above — adapt the values to your own cluster.

`experiments/cluster.env`:
```bash
export REMOTE_DIR="/scratch-shared/$USER/thesis"
export SLURM_PARTITION="mcs.gpu.q"
export SLURM_GRES="gpu:nvidia_geforce_rtx_2080_ti:1"
```

- **Scratch purge.** `/scratch-shared` deletes anything older than **14 days**
  by mtime; the master copy lives in `$HOME` (200 GiB, permanent) and is staged
  with `cp` for a fresh clock. The venv and uv cache also live on scratch, so
  they purge on the same clock — rebuilding after an idle fortnight is expected.
- **uv on the shared filesystem.** Put `UV_LINK_MODE=copy` and
  `UV_CACHE_DIR=/scratch-shared/$USER/.uv-cache` in `~/.bashrc` (verify with
  `echo "$UV_LINK_MODE $UV_CACHE_DIR"`): the default hardlink mode silently
  leaves a **broken, empty torch** (`torch.__file__` is `None`). Copy mode builds
  a self-contained `.venv` (~5.5 G).
- **Windows / transfers.** `deploy.sh` / `fetch.sh` / `rsync` run from **WSL**
  (`sudo apt install rsync`); set up SSH-key auth and the host alias in WSL
  `~/.ssh`, and put `networkingMode=mirrored` in `~/.wslconfig` (required for
  rsync over the TU/e VPN; not needed on campus WiFi). Local Python
  (`collect_results`, `analyze`) can run from PowerShell. Do **not** use
  `cmd.exe` for the `ssh`/transfer lines.
- **VPN.** Needed when off-campus.
