# What a run saves, and what it's for

Everything below lives under a run's result folder
(`<LOG>/<model_string>/`, name encodes seed / λ / impl / subset). Concise map of
artifact → contents → use in the comparison.

| Artifact | Contents | Used for |
|---|---|---|
| `backup_results.csv` | Per-(validated-)epoch: train losses (composite, act, ttne, rrt, outcome) and validation metrics (MAE-TTNE, 1−DL, MAE-RRT, outcome CE/acc/macro-F1/…). Plus axiom terms: `ltn_ax{1,2}_term` (=1−sat), `ltn_ax{1,2}_contrib` (=λ·term), `ltn_ax2_residual_mass`, `ltn_ax2_min_survival`, and for `guarded_last` `ltn_ax2_min_guard` (the epoch's smallest batch-mean guard weight G; near 0 means a batch had almost no determining mass). | Convergence curves; best-epoch selection; **λ contribution share** = `ltn_ax*_contrib / composite loss` (calibration). |
| `compute_cost.csv` | Per-epoch `train_seconds`, `peak_gpu_mem_mb` (training only), grad norms `grad_norm_orig_{mean,max}` / `grad_norm_clipped_mean`, and **per-head** `grad_norm_head_{act,ttne,rrt,outcome}_mean`. | **Compute-cost axis** (efficiency) and **optimization stability** (blow-ups; which head the gradient reaches — the detach/saturation story). |
| `model_epoch_{N}.pt` | Checkpoint every epoch (model + optimizer state). | Reload the best epoch for test-set inference; audits. |
| `TEST_SET_RESULTS/averaged_results_IB.pkl` / `_CB.pkl` | Test-set **predictive** metrics, instance- and case-based: MAE-TTNE (min), DL-sim, MAE-RRT (min), multiclass outcome CE / accuracy / macro-F1 / weighted-F1 / precision / recall. | **Does the axiom help the task?** (predictive axis). |
| `TEST_SET_RESULTS/consistency_diagnostics.pkl` | Test-set **impl-agnostic consistency yardstick**. Ax1, with the summed TTNE suffix up to the predicted END (the last window position when none is emitted): `mean_abs_gap`, `rmse_gap`, `mean_signed_gap`, `signed_bias_{rrt,ttne_sum}`, `mae_{rrt,ttne_sum}`, `error_correlation`, `frac_neg_*`, `frac_no_pred_end`; `mean_abs_gap_capped` / `mae_ttne_sum_capped` cut the sum at the earlier of the predicted and the true END (uses the label). Ax2: `outcome_{head,suffix}_accuracy`, `head_vs_suffix_agreement`, `disagreement_rate` (split into `conflict_rate` and `no_determining_rate`), `outcome_{head,suffix}_macro_f1`, `outcome_agreement_macro_f1` (head vs suffix-implied class), `frac_suffix_no_determining_act` (IB & CB). | **Axiom satisfaction / consistency** scored identically across impls — the shared yardstick. |
| `TEST_SET_RESULTS/consistency_per_instance.pt` | **Raw** per-instance quantities behind the yardstick (always saved with the diagnostics). Ax1 (all rows, seconds, clamped ≥0): `ax1_{rt,sum_ts}_{pred,true}`, `ax1_sum_ts_pred_capped`, `pred_len`, plus `dam_lev_similarity`, `pref_len`, `suf_len`. Ax2 (non-leaky rows): `ax2_head_logits` (full logits → confidence), `ax2_implied`, `ax2_has_det`, `ax2_outcome_labels`, `ax2_retain_bool_out` (maps ax2 rows back onto the ax1 rows). | **Inconsistency ↔ error at inference** (RQ step 4): recompute any error/agreement/confidence metric offline; per-instance distributions; significance tests. |
| `TEST_SET_RESULTS/` raw prediction tensors | (only with `store_preds`, also in `VAL_SET_RESULTS/`) decoded activity suffixes, TTNE suffixes (standardized, before padding after END), RRT, outcome logits, prefix and suffix lengths. | Enforcement after inference; alternative sequence metrics; error analysis. |
| `VAL_SET_RESULTS/` | The consistency diagnostics and per-instance file of the selected checkpoint on the validation set; written by every run. | Choosing λ and the enforcement direction (never on test). |
| `TRAIN_SET_RESULTS/` | The same on the training split (only with `eval_train`). | Train vs held-out inconsistency. |
| `provenance.txt` | One block per start of the run (by `experiments.run`), appended so reruns and re-evaluations keep the history: start time, sweep and index, SLURM job, host, GPU, the deployed code version (`DEPLOYED_VERSION.txt`; git for a local run) and the config. Directly in the top-level run folder. | Tracing a result back to the exact commit that produced it. |
| TensorBoard logs | Live training scalars. | Quick eyeballing during a run. |

Notes:
- Predictive + consistency test metrics are the cross-impl comparison; the term/
  contrib curves are for **λ calibration**, not for comparing impls (scales differ).
- `compute_cost.csv` is kept separate from `backup_results.csv` so the latter
  stays reproducible (timings are non-deterministic).
- Grad norms: whole-model pre-clip norm (mean + max per epoch) plus per-output-head
  means, so you can see how much gradient reaches each head under each detach mode.
- `collect_results` turns all this into two tidy tables: a per-run **summary**
  (`collect_sweeps`) and per-epoch **curves** (`collect_history`) — see the
  experiments README.
