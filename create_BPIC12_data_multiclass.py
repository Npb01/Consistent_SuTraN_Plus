"""Construct the BPIC12 tensor datasets for SuTraN+.

BPI Challenge 2012 (loan applications at a Dutch financial institute) supports
all four prediction tasks: activity suffix, time-to-next-event, remaining time,
and a 3-class case outcome.

Outcome
    Multiclass {Approved: 0, Cancelled: 1, Declined: 2}, given by the
    application-level resolution event {A_APPROVED, A_CANCELLED, A_DECLINED}
    reached by the case. Every labelable case contains exactly one of these
    three activities, so the label is unambiguous. A_APPROVED marks a granted
    loan; A_ACCEPTED is a distinct, earlier activity (application accepted into
    processing) and is not an outcome. Cases without a resolution event are
    dropped.

Lifecycle
    W_ work items carry SCHEDULE/START/COMPLETE transitions; A_ and O_ events
    are COMPLETE-only. Each work item is collapsed to its COMPLETE event, so the
    activity sequence and all time targets are those of the COMPLETE-only log.
    The START->COMPLETE handling time (`proc_time`) and SCHEDULE->START queue
    time (`wait_time`) are kept as numeric event features. Both describe already
    completed prefix events and therefore do not leak the time targets.

Input
    `BPIC12.csv` in the repository root (converted from the 4TU XES log), with
    columns case:concept:name, concept:name, time:timestamp,
    lifecycle:transition, org:resource, case:AMOUNT_REQ.

The empirical parameters (window_size, max_days, end_date, and median_caselen
in log_configs.py) can be recomputed with `compute_bpic12_params.py`.
"""

import os
import pandas as pd
import numpy as np
import torch

from Preprocessing.from_log_to_tensors import log_to_tensors

CSV_PATH = r'BPIC12.csv'

# Lifecycle ordering for breaking equal-timestamp ties within a case.
_LC_RANK = {'SCHEDULE': 0, 'START': 1, 'COMPLETE': 2}


def sort_log(df, case_id='case:concept:name', timestamp='time:timestamp', act_label='concept:name'):
    """Sort events so earlier-starting cases come first, events within a case
    ordered by timestamp. Identical to the BPIC17/BPIC19 scripts."""
    df_help = df.sort_values([case_id, timestamp], ascending=[True, True], kind='mergesort').copy()
    df_first = df_help.drop_duplicates(subset=case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending=True, kind='mergesort')
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis=1)
    df = df.merge(df_first, on=case_id, how='left')
    df = df.sort_values(['case_id_int', timestamp], ascending=[True, True], kind='mergesort')
    df = df.drop('case_id_int', axis=1)
    return df.reset_index(drop=True)


def load_bpic12_csv(path=CSV_PATH):
    """Load the BPIC12 CSV with correctly typed columns.

    Case ids and resource ids are numeric codes in BPIC12 and are read as
    strings: the preprocessing pipeline builds prefix ids by string
    concatenation, and resources are categorical. The log spans a DST change,
    so timestamps carry mixed UTC offsets (+01:00 / +02:00) and are parsed
    directly to UTC.

    Parameters
    ----------
    path : str, optional
        Path to the CSV. By default `CSV_PATH`.

    Returns
    -------
    pandas.DataFrame
        Event log with a UTC datetime64 `time:timestamp` column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} not found. Convert the BPIC12 XES log to CSV with columns "
            "case:concept:name, concept:name, time:timestamp, "
            "lifecycle:transition, org:resource, case:AMOUNT_REQ."
        )
    df = pd.read_csv(path, header=0,
                     dtype={'case:concept:name': str, 'org:resource': str})
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed', utc=True)
    return df


def collapse_lifecycle(log):
    """Collapse the W_ work-item lifecycle to one row per activity instance.

    Every COMPLETE event is kept. Within a case, a segment runs from the event
    after the previous COMPLETE up to and including the next COMPLETE; the last
    SCHEDULE and the last START in that segment give the work item's times:

        proc_time = COMPLETE - START     (seconds, handling time)
        wait_time = START - SCHEDULE     (seconds, queue time)

    A_ and O_ events carry no SCHEDULE/START and get proc_time = wait_time = 0.
    SCHEDULE/START events with no later COMPLETE in their case are dropped.

    Pairing by segment assumes a case handles one work item at a time, which is
    the normal BPIC12 pattern. Work items open concurrently within one case are
    paired by recency rather than by work-item identity.

    Parameters
    ----------
    log : pandas.DataFrame
        Event log sorted by `sort_log`, with a UTC datetime64 timestamp column.

    Returns
    -------
    pandas.DataFrame
        COMPLETE events only, in case order, with `proc_time` and `wait_time`.
    """
    # Keep the case order of sort_log; within a case, order equal timestamps
    # SCHEDULE < START < COMPLETE.
    log = log.assign(_case_pos=pd.factorize(log['case:concept:name'])[0],
                     _lc=log['lifecycle:transition'].map(_LC_RANK).fillna(1))
    log = log.sort_values(['_case_pos', 'time:timestamp', '_lc'], kind='mergesort')

    lifecycle = log['lifecycle:transition']
    is_complete = lifecycle.eq('COMPLETE')

    # A segment starts at the first event of a case and after every COMPLETE.
    new_case = log['_case_pos'].ne(log['_case_pos'].shift())
    segment = (new_case | is_complete.shift(fill_value=False)).cumsum()

    timestamps = log['time:timestamp']
    sched_ts = timestamps.where(lifecycle.eq('SCHEDULE')).groupby(segment).transform('last')
    start_ts = timestamps.where(lifecycle.eq('START')).groupby(segment).transform('last')

    out = log.loc[is_complete].drop(columns=['_case_pos', '_lc'])
    start_ts, sched_ts = start_ts[is_complete], sched_ts[is_complete]
    out['proc_time'] = (out['time:timestamp'] - start_ts).dt.total_seconds().fillna(0.0).clip(lower=0.0)
    out['wait_time'] = (start_ts - sched_ts).dt.total_seconds().fillna(0.0).clip(lower=0.0)
    return out.reset_index(drop=True)


def preprocess_bpic12(log):
    """Preprocess the BPIC12 event log into a 3-class outcome log.

    Collapses each W_ work item to its COMPLETE event, attaching `proc_time`
    and `wait_time` as numeric event features, then derives the multiclass
    outcome label and the outcome-mask index.

    Parameters
    ----------
    log : pandas.DataFrame
        Event log, sorted by `sort_log`, with a UTC datetime64 timestamp column.

    Returns
    -------
    log : pandas.DataFrame
        Preprocessed event log with one row per activity instance.
    """
    log['case:AMOUNT_REQ'] = pd.to_numeric(log['case:AMOUNT_REQ'], errors='coerce')

    log = collapse_lifecycle(log)

    # Zero-based event index within each case (every row is a COMPLETE event).
    log['event_index'] = log.groupby('case:concept:name').cumcount()

    # Application-level resolution events defining the outcome. A_APPROVED marks
    # a granted loan; A_ACCEPTED (acceptance into processing) is not an outcome.
    relevant_app_events = ['A_APPROVED', 'A_CANCELLED', 'A_DECLINED']

    # BPIC12 has no EventOrigin column, and the last A_ event of a granted loan is
    # A_ACTIVATED or A_REGISTERED rather than the resolution event. The outcome is
    # therefore read from the last resolution event, and its event index is the
    # outcome-mask index.
    log_res = log[log['concept:name'].isin(relevant_app_events)]
    last_res = log_res.groupby('case:concept:name', sort=False).last().reset_index()
    last_res = last_res[['case:concept:name', 'concept:name', 'event_index']]
    last_res.columns = ['case:concept:name', 'last_a_act', 'outc_mask_idx']

    log = log.merge(last_res, on='case:concept:name', how='left')

    # Drop cases that never reach a resolution event (no derivable label).
    log = log[~log['last_a_act'].isna()]

    # Map to the 3 classes. This index order MUST match
    # outcome_determining_activities_dict['BPIC_12'] in log_configs.py.
    out_mapping_dict = {'A_APPROVED': 0, 'A_CANCELLED': 1, 'A_DECLINED': 2}
    log['case:outcome_int'] = log['last_a_act'].map(out_mapping_dict).astype('int32')

    log = log.drop(['event_index', 'last_a_act'], axis=1)
    return log


def construct_BPIC12_datasets():
    df = load_bpic12_csv()
    df = sort_log(df)
    df = preprocess_bpic12(df)

    # proc_time and wait_time are derived in preprocess_bpic12.
    categorical_casefeatures = []
    numeric_casefeatures = ['case:AMOUNT_REQ']
    categorical_eventfeatures = ['org:resource']
    numeric_eventfeatures = ['proc_time', 'wait_time']
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'

    # Empirical parameters, measured on the preprocessed log by
    # compute_bpic12_params.py.
    start_date = None
    # The log ends on 2012-03-14; March holds only a partial, right-censored
    # tail of case completions, which is excluded.
    end_date = "2012-02"
    start_before_date = None
    max_days = 40.17         # 98.5th percentile of case duration (days)
    window_size = 45         # 98.5th percentile of case length (events)

    log_name = 'BPIC_12'
    outcome = 'case:outcome_int'
    outcome_type = 'multiclass_outcome'
    outcome_mask = 'outc_mask_idx'
    test_len_share = 0.25
    val_len_share = 0.2
    mode = 'preferred'

    train_tensors, val_tensors, test_tensors = log_to_tensors(
        df,
        log_name=log_name,
        start_date=start_date,
        start_before_date=start_before_date,
        end_date=end_date,
        max_days=max_days,
        test_len_share=test_len_share,
        val_len_share=val_len_share,
        window_size=window_size,
        mode=mode,
        case_id=case_id,
        act_label=act_label,
        timestamp=timestamp,
        cat_casefts=categorical_casefeatures,
        num_casefts=numeric_casefeatures,
        cat_eventfts=categorical_eventfeatures,
        num_eventfts=numeric_eventfeatures,
        outcome=outcome,
        outcome_type=outcome_type,
        outcome_mask=outcome_mask,
    )

    train_data, og_caseint_train = train_tensors[0], train_tensors[1]
    val_data, og_caseint_val = val_tensors[0], val_tensors[1]
    test_data, og_caseint_test = test_tensors[0], test_tensors[1]

    if outcome and outcome_mask:
        instance_mask_out_train = train_tensors[-1]
        instance_mask_out_val = val_tensors[-1]
        instance_mask_out_test = test_tensors[-1]

    output_directory = log_name
    os.makedirs(output_directory, exist_ok=True)

    torch.save(train_data, os.path.join(output_directory, 'train_tensordataset.pt'))
    torch.save(og_caseint_train, os.path.join(output_directory, 'og_caseint_train.pt'))
    torch.save(val_data, os.path.join(output_directory, 'val_tensordataset.pt'))
    torch.save(og_caseint_val, os.path.join(output_directory, 'og_caseint_val.pt'))
    torch.save(test_data, os.path.join(output_directory, 'test_tensordataset.pt'))
    torch.save(og_caseint_test, os.path.join(output_directory, 'og_caseint_test.pt'))

    if outcome and outcome_mask:
        torch.save(instance_mask_out_train, os.path.join(output_directory, 'instance_mask_out_train.pt'))
        torch.save(instance_mask_out_val, os.path.join(output_directory, 'instance_mask_out_val.pt'))
        torch.save(instance_mask_out_test, os.path.join(output_directory, 'instance_mask_out_test.pt'))


def main():
    construct_BPIC12_datasets()


if __name__ == '__main__':
    main()
