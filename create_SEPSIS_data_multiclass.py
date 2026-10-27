"""Construct the SEPSIS tensor datasets for SuTraN+.

Sepsis Cases (Mannhardt, 4TU.ResearchData): trajectories of patients with
suspected sepsis in a Dutch hospital, from emergency-room registration to
discharge. The log supports all four prediction tasks: activity suffix,
time-to-next-event, remaining time, and a two-class case outcome (trained
with the multiclass outcome head, so axiom 2 applies).

Outcome
    Discharge type, two classes: {Release A: 0, other release: 1}, the latter
    covering Release B, C, D and E (the labelling of Teinemaa et al.'s
    sepsis_3 task). Every released case contains exactly one release event, so
    the label is unambiguous; cases without a release (still admitted when the
    log was extracted) are dropped. For axiom 2, class 1 is implied by any of
    its four release activities.

Case boundary
    A case ends at its release event: one case is one hospital stay. Events
    after discharge (almost exclusively Return ER, which can follow weeks or
    months later) are removed, so remaining-time targets describe the stay
    itself. The release is therefore the last event of every case, and the
    outcome mask only excludes the full-length prefix.

Features
    The checklist booleans and Age are recorded on the ER Registration event and
    are lifted to case attributes. Diagnose (about 145 mostly unique codes) is
    not used. Lab values (CRP, Leucocytes, LacticAcid) are recorded on their own
    lab events; they are forward-filled per case, so every event carries the
    last value measured so far. Events before the first measurement stay
    missing and receive the pipeline's missing-value indicator.

Split
    Timestamps in this log were randomised per case (gaps within a case are
    preserved), so the chronological train/test split of the pipeline amounts
    to a fixed random partition of cases, and date filters are not used.

Data quirks
    One case id is the literal string "NA", so the CSV is read without pandas'
    default NA parsing. Exact duplicate rows are dropped before event indices
    are computed, so the outcome-mask index refers to the final sequence. About
    half of all events share their timestamp with another event of the same
    case; their order follows the event order of the log.

Input
    `SEPSIS.csv` in the repository root, converted from the XES log with
    `Preprocessing/xes_to_csv.py`. The empirical parameters (window_size,
    max_days, and median_caselen in log_configs.py) can be recomputed with
    `compute_sepsis_params.py`.
"""

import os

import pandas as pd
import torch

from Preprocessing.from_log_to_tensors import log_to_tensors

CSV_PATH = r'SEPSIS.csv'

CASE_ID = 'case:concept:name'
ACTIVITY = 'concept:name'
TIMESTAMP = 'time:timestamp'

RELEASE_A = 'Release A'
OTHER_RELEASES = ['Release B', 'Release C', 'Release D', 'Release E']
OUT_MAPPING = {RELEASE_A: 0, **{r: 1 for r in OTHER_RELEASES}}

# Attributes recorded on the ER Registration event, lifted to case level.
CHECKLIST_ATTRIBUTES = [
    'InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea',
    'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup',
    'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
    'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature',
    'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie',
    'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment',
    'DiagnosticECG',
]
NUMERIC_CASE_ATTRIBUTES = ['Age']
LAB_VALUES = ['CRP', 'Leucocytes', 'LacticAcid']


def load_sepsis_csv(path=CSV_PATH):
    """Load the SEPSIS CSV with correctly typed columns.

    Every column is read as a string and only empty fields count as missing:
    the case ids are letter codes and include the literal "NA". Timestamps are
    parsed to UTC, Age and the lab values to numbers.

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
            f"{path!r} not found. Convert the Sepsis Cases XES log with "
            "`python Preprocessing/xes_to_csv.py <log.xes> SEPSIS.csv`."
        )
    df = pd.read_csv(path, header=0, dtype=str, keep_default_na=False, na_values=[''])
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], format='mixed', utc=True)
    for col in NUMERIC_CASE_ATTRIBUTES + LAB_VALUES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def sort_log(df, case_id=CASE_ID, timestamp=TIMESTAMP, act_label=ACTIVITY):
    """Sort events so earlier-starting cases come first, events within a case
    ordered by timestamp (stable, so ties keep the log's event order).
    Identical to the other builders."""
    df_help = df.sort_values([case_id, timestamp], ascending=[True, True], kind='mergesort').copy()
    df_first = df_help.drop_duplicates(subset=case_id)[[case_id, timestamp]].copy()
    df_first = df_first.sort_values(timestamp, ascending=True, kind='mergesort')
    df_first['case_id_int'] = [i for i in range(len(df_first))]
    df_first = df_first.drop(timestamp, axis=1)
    df = df.merge(df_first, on=case_id, how='left')
    df = df.sort_values(['case_id_int', timestamp], ascending=[True, True], kind='mergesort')
    df = df.drop('case_id_int', axis=1)
    return df.reset_index(drop=True)


def preprocess_sepsis(log):
    """Preprocess the Sepsis log into a 2-class discharge-type outcome log.

    Parameters
    ----------
    log : pandas.DataFrame
        Event log sorted by `sort_log`, as returned by `load_sepsis_csv`.

    Returns
    -------
    pandas.DataFrame
        One row per event up to and including the release, with case-level
        attributes `case:<name>`, forward-filled lab values, the outcome
        `case:outcome_int` and the outcome-mask index `outc_mask_idx`.
    """
    by_case = log.groupby(CASE_ID, sort=False)

    # Registration attributes -> case attributes (first recorded value).
    for col in CHECKLIST_ATTRIBUTES + NUMERIC_CASE_ATTRIBUTES:
        log['case:' + col] = by_case[col].transform('first')
    log = log.drop(columns=CHECKLIST_ATTRIBUTES + NUMERIC_CASE_ATTRIBUTES + ['Diagnose'])

    # Last measured lab value so far, per case.
    for col in LAB_VALUES:
        log[col] = log.groupby(CASE_ID, sort=False)[col].ffill()

    # Cut every case after its release event; drop cases without one.
    position = log.groupby(CASE_ID, sort=False).cumcount()
    is_release = log[ACTIVITY].isin(OUT_MAPPING)
    release_pos = position.where(is_release).groupby(log[CASE_ID], sort=False).transform('max')
    log = log[release_pos.notna() & (position <= release_pos)]

    log = log.drop_duplicates(ignore_index=True)

    # Outcome and mask index on the final event sequence. The release is the
    # last event, so its index is the case length minus one.
    release_act = log[ACTIVITY].where(log[ACTIVITY].isin(OUT_MAPPING))
    by_case = log.groupby(CASE_ID, sort=False)
    log['case:outcome_int'] = (release_act.groupby(log[CASE_ID], sort=False)
                               .transform('last').map(OUT_MAPPING).astype('int32'))
    log['outc_mask_idx'] = by_case[ACTIVITY].transform('size') - 1
    return log


def construct_SEPSIS_datasets():
    df = load_sepsis_csv()
    df = sort_log(df)
    df = preprocess_sepsis(df)

    categorical_casefeatures = ['case:' + c for c in CHECKLIST_ATTRIBUTES]
    numeric_casefeatures = ['case:' + c for c in NUMERIC_CASE_ATTRIBUTES]
    categorical_eventfeatures = ['org:group']
    numeric_eventfeatures = LAB_VALUES

    # Empirical parameters, measured on the preprocessed log by
    # compute_sepsis_params.py. No date filters: timestamps are randomised.
    start_date = None
    end_date = None
    start_before_date = None
    max_days = 29.86         # 98.5th percentile of case duration (days)
    window_size = 47         # 98.5th percentile of case length (events)

    log_name = 'SEPSIS'
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
        case_id=CASE_ID,
        act_label=ACTIVITY,
        timestamp=TIMESTAMP,
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
    torch.save(instance_mask_out_train, os.path.join(output_directory, 'instance_mask_out_train.pt'))
    torch.save(instance_mask_out_val, os.path.join(output_directory, 'instance_mask_out_val.pt'))
    torch.save(instance_mask_out_test, os.path.join(output_directory, 'instance_mask_out_test.pt'))


def main():
    construct_SEPSIS_datasets()


if __name__ == '__main__':
    main()
