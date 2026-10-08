# What a run saves, and what it's for

Everything below lives under a run's result folder
(`<LOG>/<model_string>/`, name encodes seed / λ / impl / subset). Concise map of
artifact → contents → use in the comparison.

| Artifact | Contents | Used for |
|---|---|---|
| `backup_results.csv` | Per-(validated-)epoch: train losses (composite, act, ttne, rrt, outcome) and validation metrics (MAE-TTNE, 1−DL, MAE-RRT, outcome CE/acc/macro-F1/…). Plus axiom terms: `ltn_ax{1,2}_term` (=1−sat), `ltn_ax{1,2}_contrib` (=λ·term), `ltn_ax2_residual_mass`, `ltn_ax2_min_survival`. | Convergence curves; best-epoch selection; **λ contribution share** = `ltn_ax*_contrib / composite loss` (calibration). |
| `compute_cost.csv` | Per-epoch `train_seconds`, `peak_gpu_mem_mb` (training only), grad norms `grad_norm_orig_{mean,max}` / `grad_norm_clipped_mean`, and **per-head** `grad_norm_head_{act,ttne,rrt,outcome}_mean`. | **Compute-cost axis** (efficiency) and **optimization stability** (blow-ups; which head the gradient reaches — the detach/saturation story). |
| `model_epoch_{N}.pt` | Checkpoint every epoch (model + optimizer state). | Reload the best epoch for test-set inference; audits. |
| `TEST_SET_RESULTS/averaged_results_IB.pkl` / `_CB.pkl` | Test-set **predictive** metrics, instance- and case-based: MAE-TTNE (min), DL-sim, MAE-RRT (min), multiclass outcome CE / accuracy / macro-F1 / weighted-F1 / precision / recall. | **Does the axiom help the task?** (predictive axis). |
| `TEST_SET_RESULTS/consistency_diagnostics.pkl` | Test-set **impl-agnostic consistency yardstick**. Ax1: `mean_abs_gap`, `mean_signed_gap`, `signed_bias_{rrt,ttne_sum}`, `mae_{rrt,ttne_sum}`, `error_correlation`, `frac_neg_*`. Ax2: `outcome_{head,suffix}_accuracy`, `head_vs_suffix_agreement`, `disagreement_rate`, `outcome_{head,suffix}_macro_f1`, `frac_suffix_no_determining_act` (all IB & CB). | **Axiom satisfaction / consistency** scored identically across impls — the shared yardstick. |
| `TEST_SET_RESULTS/consistency_per_instance.pt` | **Raw** per-instance quantities behind the yardstick (always saved with the diagnostics). Ax1 (all rows, seconds, clamped ≥0): `ax1_{rt,sum_ts}_{pred,true}`, plus `dam_lev_similarity`, `pref_len`, `suf_len`. Ax2 (non-leaky rows): `ax2_head_logits` (full logits → confidence), `ax2_implied`, `ax2_has_det`, `ax2_outcome_labels`, `ax2_retain_bool_out` (maps ax2 rows back onto the ax1 rows). | **Inconsistency ↔ error at inference** (RQ step 4): recompute any error/agreement/confidence metric offline; per-instance distributions; significance tests. |
| `TEST_SET_RESULTS/` raw prediction tensors | (only when `store_preds`, off by default) full decoded suffixes / out-preds / lengths. | Alternative sequence metrics; deep error analysis. |
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
