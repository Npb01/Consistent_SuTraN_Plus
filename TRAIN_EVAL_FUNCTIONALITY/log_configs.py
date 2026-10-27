"""
Module containing the event log (dataset) properties 
for the event logs used in the final experimental setup of the SuTraN+
paper.
"""

# The log names are:
log_name_list = ['BPIC_19', 'BPIC_17_DR', 'BPIC_17', 'BPIC_12', 'SEPSIS']


# Defining the log-specific parameters in dictionaries
#   median_caselen : the median case length of the log
#   BPIC_12: measured via compute_bpic12_params.py on the resolved,
#   duration-collapsed log.
#   SEPSIS: measured via compute_sepsis_params.py on the cut-at-release log.
median_caselen_dict = {'BPIC_19': 5,
                       'BPIC_17_DR': 21,
                       'BPIC_17': 34,
                       'BPIC_12': 6,
                       'SEPSIS': 14}

#   outcome_bool
outcome_bools_dict = {'BPIC_19': False,
                      'BPIC_17_DR': True,
                      'BPIC_17': True,
                      'BPIC_12': True,
                      'SEPSIS': True}

#   out_mask
out_masks_dict = {'BPIC_19' : False,
                  'BPIC_17_DR' : True,
                  'BPIC_17' : True,
                  'BPIC_12' : True,
                  'SEPSIS' : True}

#   out_type
out_types_dict = {'BPIC_19' : None,
                  'BPIC_17_DR' : 'multiclass_outcome',
                  'BPIC_17' : 'multiclass_outcome',
                  'BPIC_12' : 'multiclass_outcome',
                  'SEPSIS' : 'multiclass_outcome'}
#   num_outclasses
num_outclasses_dict = {'BPIC_19' : None,
                       'BPIC_17_DR' : 3,
                       'BPIC_17' : 3,
                       'BPIC_12' : 3,
                       'SEPSIS' : 2}

#   outcome_determining_activities (axiom 2)
#   Maps each outcome class index to the activity (or list of activities) whose
#   LAST occurrence in the suffix implies that outcome. `None` for logs without
#   an outcome head.
#
#   Activity NAMES, not integer ids: the tensor ids are `categ_mapping + 1` and
#   are resolved at runtime by `outcome_consistency_metrics.resolve_determining_ids`.
#   Hardcoding ids would keep "working" silently against a re-preprocessed log
#   with a different vocabulary; names raise KeyError instead.
#
#   Class indices come from the preprocessing scripts
#   (create_BPIC17_DR_data_multiclass.py): {'Accepted': 0, 'Canceled': 1,
#   'Refused': 2}. The LAST-occurrence reading is the only exact one -- an
#   any-occurrence reading of O_Accepted holds only ~70% of the time, because an
#   accepted offer can still be cancelled afterwards.
#   BPIC_12: index order matches `out_mapping_dict` in
#   create_BPIC12_data_multiclass.py: {'A_APPROVED': 0, 'A_CANCELLED': 1,
#   'A_DECLINED': 2}. A_APPROVED marks a granted loan (A_ACCEPTED is an earlier,
#   non-terminal activity). Every labelable case contains exactly one of the
#   three resolution activities, so both the last- and the any-occurrence
#   readings are exact for this log.
#   SEPSIS: index order matches `OUT_MAPPING` in create_SEPSIS_data_multiclass.py
#   ({'Release A': 0, other release: 1}). Every case contains exactly one
#   release event (it is the last event of the case), so both readings are
#   exact; class 1 is implied by any of Release B-E.
outcome_determining_activities_dict = {'BPIC_19' : None,
                                       'BPIC_17_DR' : {0: 'O_Accepted',
                                                       1: 'O_Cancelled',
                                                       2: 'O_Refused'},
                                       'BPIC_17' : {0: 'O_Accepted',
                                                    1: 'O_Cancelled',
                                                    2: 'O_Refused'},
                                       'BPIC_12' : {0: 'A_APPROVED',
                                                    1: 'A_CANCELLED',
                                                    2: 'A_DECLINED'},
                                       'SEPSIS' : {0: 'Release A',
                                                   1: ['Release B', 'Release C',
                                                       'Release D', 'Release E']}}
