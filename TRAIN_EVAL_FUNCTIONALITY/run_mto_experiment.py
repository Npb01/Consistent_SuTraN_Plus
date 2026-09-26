"""
Entry-point for reproducing the SuTraN+ multi-task experiments.

This wrapper exposes a tiny, user-facing API that only requires the
event log identifier, the desired MTO technique, and the seed. All
other log-specific and technique-specific hyperparameters are pulled
from `log_configs.py` and `technique_configs.py`, matching the settings
reported in the paper.
"""

from __future__ import annotations

import argparse
import inspect
from typing import Callable, Dict

from TRAIN_EVAL_FUNCTIONALITY import log_configs, technique_configs
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_EQUAL_WEIGHTING import (
    train_eval as train_eval_equal_weighting,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_GRADNORM import (
    train_eval as train_eval_gradnorm,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_PCGRAD import (
    train_eval as train_eval_pcgrad,
)
from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_UW import (
    train_eval as train_eval_uw,
)

# ---------------------------------------------------------------------------
# Technique dispatch and configuration
# ---------------------------------------------------------------------------

_TECHNIQUE_DISPATCH: Dict[str, Callable[..., None]] = {
    "equal_weighting": train_eval_equal_weighting,
    "gradnorm": train_eval_gradnorm,
    "uw": train_eval_uw,
    "uw_plus": train_eval_uw,
    "pcgrad": train_eval_pcgrad,
}

_TECHNIQUE_CONFIGS: Dict[str, Dict[str, object]] = {
    "equal_weighting": {},
    "gradnorm": technique_configs.gradnorm_config_dict,
    "uw": technique_configs.uw_config_dict,
    "uw_plus": technique_configs.uw_plus_config_dict,
    "pcgrad": {"lr_model": 0.0002},
}


def run_mto_experiment(
    log_name: str,
    mto_technique: str,
    seed: int,
    subset_fraction: float = 1.0,
    val_subset_fraction: float = 1.0,
    num_epochs: int = 200,
    patience: int = 24,
    lambda_ltn: float = 0.0,
    detach_mode: str = "none",
) -> None:
    """
    Train and evaluate SuTraN+ with the specified MTO technique.

    Parameters
    ----------
    log_name : {'BPIC_17', 'BPIC_17_DR', 'BPIC_19'}
        Identifier of the event log to use.
    mto_technique : {'equal_weighting', 'gradnorm', 'uw', 'uw_plus', 'pcgrad'}
        Multi-task optimisation strategy to deploy.
    seed : int
        Random seed (1..5 in the paper experiments).
    subset_fraction, val_subset_fraction : float
        Case-level subset fractions for the train and validation splits,
        in (0, 1]. Forwarded only to techniques whose ``train_eval`` accepts
        them (equal_weighting, uw); ignored by the others.
    num_epochs : int
        Maximum number of training epochs. By default 200.
    patience : int
        Early-stopping patience in epochs. By default 24.
    lambda_ltn : float
        Weight of the axiom-1 time-consistency term (0.0 = off). Forwarded
        only to techniques whose ``train_eval`` accepts it (equal_weighting, uw).
    detach_mode : {'none', 'ttne', 'rrt'}
        Which side of the axiom-1 term receives gradient. By default 'none'.
    """

    log_key = log_name.upper()
    if log_key not in log_configs.log_name_list:
        raise ValueError(
            f"'log_name' must be one of {log_configs.log_name_list}, got '{log_name}'."
        )

    technique_key = mto_technique.lower()
    if technique_key not in _TECHNIQUE_DISPATCH:
        raise ValueError(
            f"Unsupported MTO technique '{mto_technique}'. "
            "Choose from 'equal_weighting', 'gradnorm', 'uw', 'uw_plus', or 'pcgrad'."
        )

    train_fn = _TECHNIQUE_DISPATCH[technique_key]
    technique_kwargs = dict(_TECHNIQUE_CONFIGS[technique_key])  # copy

    # Gather log-specific parameters.
    base_kwargs = {
        "log_name": log_key,
        "median_caselen": log_configs.median_caselen_dict[log_key],
        "outcome_bool": log_configs.outcome_bools_dict[log_key],
        "out_mask": log_configs.out_masks_dict[log_key],
        "clen_dis_ref": True,
        "out_type": log_configs.out_types_dict[log_key],
        "num_outclasses": log_configs.num_outclasses_dict[log_key],
        "seed": seed,
        "out_string": None,
    }

    base_kwargs.update(technique_kwargs)

    # Experiment-control kwargs are only supported by some techniques
    # (equal_weighting, uw). Forward each only if train_fn accepts it, so the
    # other techniques keep their original signatures.
    experiment_kwargs = {
        "subset_fraction": subset_fraction,
        "val_subset_fraction": val_subset_fraction,
        "num_epochs": num_epochs,
        "patience": patience,
        "lambda_ltn": lambda_ltn,
        "detach_mode": detach_mode,
    }
    accepted = inspect.signature(train_fn).parameters
    base_kwargs.update({k: v for k, v in experiment_kwargs.items() if k in accepted})

    train_fn(**base_kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate SuTraN+ with a selected multi-task optimisation technique."
        )
    )
    parser.add_argument(
        "--log_name",
        required=True,
        choices=log_configs.log_name_list,
        help="Event log identifier.",
    )
    parser.add_argument(
        "--MTO_technique",
        required=True,
        choices=list(_TECHNIQUE_DISPATCH.keys()),
        help="Multi-task optimisation strategy to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (1..5 in the paper experiments).",
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of training cases to use, in (0, 1].",
    )
    parser.add_argument(
        "--val_subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of validation cases for per-epoch validation, in (0, 1].",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=24,
        help="Early-stopping patience in epochs.",
    )
    parser.add_argument(
        "--lambda_ltn",
        type=float,
        default=0.0,
        help="Weight of the axiom-1 time-consistency term (0.0 = off).",
    )
    parser.add_argument(
        "--detach_mode",
        type=str,
        default="none",
        choices=["none", "ttne", "rrt"],
        help="Which side of the axiom-1 term to detach ('none', 'ttne', 'rrt').",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_arg_parser().parse_args()
    run_mto_experiment(
        log_name=arguments.log_name,
        mto_technique=arguments.MTO_technique,
        seed=arguments.seed,
        subset_fraction=arguments.subset_fraction,
        val_subset_fraction=arguments.val_subset_fraction,
        num_epochs=arguments.num_epochs,
        patience=arguments.patience,
        lambda_ltn=arguments.lambda_ltn,
        detach_mode=arguments.detach_mode,
    )
