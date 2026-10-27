"""Compute the empirical SEPSIS preprocessing parameters.

Prints the log-specific numbers used by the dataset build:

  * window_size    -> create_SEPSIS_data_multiclass.py
  * max_days       -> create_SEPSIS_data_multiclass.py
  * median_caselen -> TRAIN_EVAL_FUNCTIONALITY/log_configs.py

No date parameters: the log's timestamps are randomised per case.

Values are measured on the frame the pipeline receives: the log after sort_log
and preprocess_sepsis (cases cut at their release, cases without a release
dropped, duplicate rows removed). Also printed: the class balance and the share
of cases the window_size / max_days filters would drop.

Usage:
    python compute_sepsis_params.py
"""

import os
import sys

from create_SEPSIS_data_multiclass import (CASE_ID, CSV_PATH, TIMESTAMP,
                                           load_sepsis_csv, preprocess_sepsis,
                                           sort_log)


def main():
    if not os.path.exists(CSV_PATH):
        print(
            f"ERROR: {CSV_PATH!r} not found in the repository root.\n"
            "Convert the Sepsis Cases XES log first: "
            "python Preprocessing/xes_to_csv.py <log.xes> SEPSIS.csv",
            file=sys.stderr,
        )
        return 1

    df = preprocess_sepsis(sort_log(load_sepsis_csv()))

    g = df.groupby(CASE_ID)
    caselen = g.size()
    dur_days = (g[TIMESTAMP].max() - g[TIMESTAMP].min()).dt.total_seconds() / 86400.0

    median_caselen = int(caselen.median())
    window_size = int(caselen.quantile(0.985))
    max_days = round(float(dur_days.quantile(0.985)), 2)
    outcome = g['case:outcome_int'].first().value_counts().sort_index()
    kept = (caselen <= window_size) & (dur_days <= max_days)

    print("SEPSIS parameters (measured on the preprocessed log):")
    print(f"  num cases              : {caselen.size}")
    print(f"  num events             : {len(df)}")
    print(f"  outcome classes        : {outcome.to_dict()}  (0 = Release A, 1 = other release)")
    print(f"  cases kept by filters  : {kept.sum()} ({kept.mean():.1%})")
    print()
    print("  create_SEPSIS_data_multiclass.py:")
    print(f"    window_size = {window_size}        # 98.5th pct case length")
    print(f"    max_days    = {max_days}     # 98.5th pct case duration (days)")
    print()
    print("  TRAIN_EVAL_FUNCTIONALITY/log_configs.py:")
    print(f"    median_caselen_dict['SEPSIS'] = {median_caselen}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
