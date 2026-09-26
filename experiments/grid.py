"""Turn a grid spec into a list of run configs, and resolve their run dirs.

A config is just a ``dict`` of keyword arguments for
``run_mto_experiment.run_mto_experiment`` -- keys map 1:1 to that function's
parameters, and anything omitted falls back to its default.
"""
from __future__ import annotations

import itertools

from TRAIN_EVAL_FUNCTIONALITY import naming, log_configs

# Config keys that affect the result-folder name. Everything else (num_epochs,
# patience, batch_size, validate_every, val_subset_fraction, ...) is invisible
# to the folder name -- so two configs that differ ONLY in those collapse to the
# same run dir. Sweep those per-sweep as fixed scalars, not across configs.
_NAME_KEYS = ("subset_fraction", "lambda_ltn", "detach_mode",
              "lambda_ltn_outcome", "detach_mode_outcome",
              "balance_losses", "scale_ttne", "scale_rrt")


def expand_grid(grid):
    """Cartesian product of a ``{param: values}`` spec -> ``list[dict]``.

    A value given as a list/tuple is swept; any other value is a fixed scalar
    applied to every config. Config order is stable.
    """
    swept = {k: list(v) for k, v in grid.items() if isinstance(v, (list, tuple))}
    fixed = {k: v for k, v in grid.items() if not isinstance(v, (list, tuple))}
    keys = list(swept.keys())
    configs = []
    for combo in itertools.product(*(swept[k] for k in keys)):
        cfg = dict(fixed)
        cfg.update(dict(zip(keys, combo)))
        configs.append(cfg)
    return configs


def resolve_run_dir(cfg):
    """The top-level result folder a config will produce.

    ``out_type`` is not a config knob -- it is fixed per log in ``log_configs``
    -- so it is looked up here rather than taken from ``cfg``.
    """
    log_key = cfg["log_name"].upper()
    name_kwargs = {k: cfg[k] for k in _NAME_KEYS if k in cfg}
    return naming.run_dir(
        log_key,
        seed=cfg["seed"],
        out_type=log_configs.out_types_dict[log_key],
        out_string=cfg.get("out_string"),
        **name_kwargs,
    )


def dedup_by_run_dir(configs):
    """Drop configs that resolve to a run dir already seen (keep first).

    This is what collapses a ``lambda_ltn=0.0`` baseline swept over several
    ``detach_mode`` values -- all of which produce the identical folder name --
    into a single baseline run, without the sweep author special-casing it.
    """
    seen, out = set(), []
    for cfg in configs:
        d = resolve_run_dir(cfg)
        if d in seen:
            continue
        seen.add(d)
        out.append(cfg)
    return out


def configs_for(sweep):
    """Final ordered config list for a Sweep: expand the grid, then dedup."""
    return dedup_by_run_dir(expand_grid(sweep.grid))
