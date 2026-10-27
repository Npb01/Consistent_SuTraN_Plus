"""Compute the empirical BPIC12 preprocessing parameters.

Prints the four log-specific numbers that must be filled in by hand before the
dataset can be built:

  * window_size   -> create_BPIC12_data_multiclass.py
  * max_days      -> create_BPIC12_data_multiclass.py
  * end_date      -> create_BPIC12_data_multiclass.py (choose from the time span)
  * median_caselen-> TRAIN_EVAL_FUNCTIONALITY/log_configs.py

Values are measured on the EXACT frame the pipeline sees: the log after
sort_log + the duration-preserving collapse + outcome-resolution dropping (so
one row per activity instance, cases without a resolution event removed).

Usage:
    python compute_bpic12_params.py
"""

import os
import sys

from create_BPIC12_data_multiclass import (CSV_PATH, load_bpic12_csv,
                                           preprocess_bpic12, sort_log)


def main():
    if not os.path.exists(CSV_PATH):
        print(
            f"ERROR: {CSV_PATH!r} not found in the repository root.\n"
            "Convert the BPIC12 XES log to CSV first, with columns: "
            "case:concept:name, concept:name, time:timestamp, "
            "lifecycle:transition, org:resource, case:AMOUNT_REQ.",
            file=sys.stderr,
        )
        return 1

    # Reproduce exactly what the pipeline sees before tensorisation.
    df = sort_log(load_bpic12_csv())
    df = preprocess_bpic12(df)

    g = df.groupby('case:concept:name')
    caselen = g.size()
    dur_days = (g['time:timestamp'].max() - g['time:timestamp'].min()).dt.total_seconds() / 86400.0

    median_caselen = int(caselen.median())
    window_size = int(caselen.quantile(0.985))
    max_days = round(float(dur_days.quantile(0.985)), 2)
    span_start = df['time:timestamp'].min()
    span_end = df['time:timestamp'].max()

    print("BPIC12 parameters (measured on the resolved, collapsed log):")
    print(f"  num cases              : {caselen.size}")
    print(f"  num events (instances) : {len(df)}")
    print(f"  time span              : {span_start}  ->  {span_end}")
    print()
    print("  Paste into create_BPIC12_data_multiclass.py:")
    print(f"    window_size = {window_size}        # 98.5th pct case length")
    print(f"    max_days    = {max_days}     # 98.5th pct case duration (days)")
    print(f"    end_date    = '...'      # pick from the time span above so last ~25% of cases -> test")
    print()
    print("  Paste into TRAIN_EVAL_FUNCTIONALITY/log_configs.py:")
    print(f"    median_caselen_dict['BPIC_12'] = {median_caselen}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
