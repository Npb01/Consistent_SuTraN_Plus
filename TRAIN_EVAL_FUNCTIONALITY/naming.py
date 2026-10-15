"""Single source of truth for experiment result-folder names.

`train_eval` builds its `model_string` from `build_model_string`, and the
experiments layer (`experiments/grid.py`) predicts a config's run directory with
the same function -- so the predicted and actual paths cannot drift apart. This
replaces the hand-mirrored `run_dir()` logic the old run_configs carried.
"""
from pathlib import Path

from axiom_builders import AXIOM1_DEFAULT, AXIOM2_DEFAULT


def build_model_string(*, seed, subset_fraction=1.0, lambda_ltn=0.0,
                        detach_mode="none", lambda_ltn_outcome=0.0,
                        detach_mode_outcome="none",
                        lambda_ltn_schedule="constant", lambda_ltn_warmup_epochs=0,
                        detach_mode_late=None, detach_switch_epoch=0,
                        lambda_ltn_outcome_schedule="constant",
                        lambda_ltn_outcome_warmup_epochs=0,
                        detach_mode_outcome_late=None, detach_switch_outcome_epoch=0,
                        axiom1_impl=AXIOM1_DEFAULT, axiom2_impl=AXIOM2_DEFAULT,
                        balance_losses=False,
                        scale_ttne=1.0, scale_rrt=1.0, batch_size=128,
                        out_type=None, out_string=None):
    """The result-folder name for one run configuration.

    This IS the definition (not a mirror of anything). Tag order matters: the
    aggregation / notebook parsers key off these substrings, so appending new
    tags is safe but reordering existing ones is not. Schedule tags are emitted
    only for non-default schedules, so constant-lambda / fixed-detach runs keep
    byte-identical names.
    """
    name = "SUTRAN_DA_results"
    if subset_fraction < 1.0:
        name += "_subset_{}".format(subset_fraction)
    if lambda_ltn > 0.0:
        name += "_ltn_{}".format(lambda_ltn)
        if detach_mode != "none":
            name += "_detach_{}".format(detach_mode)
        if lambda_ltn_schedule != "constant":
            name += "_sch_{}_{}".format(lambda_ltn_schedule, lambda_ltn_warmup_epochs)
        if detach_mode_late is not None:
            name += "_dsch_{}_sw{}".format(detach_mode_late, detach_switch_epoch)
        if axiom1_impl != AXIOM1_DEFAULT:
            name += "_ax1impl_{}".format(axiom1_impl)
    if lambda_ltn_outcome > 0.0:
        name += "_ltnout_{}".format(lambda_ltn_outcome)
        if detach_mode_outcome != "none":
            name += "_detachout_{}".format(detach_mode_outcome)
        if lambda_ltn_outcome_schedule != "constant":
            name += "_schout_{}_{}".format(
                lambda_ltn_outcome_schedule, lambda_ltn_outcome_warmup_epochs)
        if detach_mode_outcome_late is not None:
            name += "_dschout_{}_sw{}".format(
                detach_mode_outcome_late, detach_switch_outcome_epoch)
        if axiom2_impl != AXIOM2_DEFAULT:
            name += "_ax2impl_{}".format(axiom2_impl)
    if balance_losses:
        name += "_balanced_ttne{}_rrt{}".format(scale_ttne, scale_rrt)
    # batch_size is otherwise invisible to the name; tag it only when it differs
    # from the default (128) so default runs keep byte-identical folder names.
    if batch_size != 128:
        name += "_bs_{}".format(batch_size)
    if out_type:
        name += "_" + out_type
        if out_string:
            name += "_" + out_string
    name += "_seed_{}".format(seed)
    return name


def run_dir(log_name, **cfg):
    """Top-level result folder for a config: ``<log_name>/<model_string>``.

    Its existence means a run was started for this config; the experiments layer
    uses it to skip completed configs and to refuse silent overwrites.
    """
    return Path(log_name) / build_model_string(**cfg)
