"""
Module containing the event log (dataset) properties 
for the event logs used in the final experimental setup of the SuTraN+
paper.
"""

# The log names are:
log_name_list = ['BPIC_19', 'BPIC_17_DR', 'BPIC_17']


# Defining the log-specific parameters in dictionaries 
#   median_caselen : the median case length of the log
median_caselen_dict = {'BPIC_19': 5, 
                       'BPIC_17_DR': 21, 
                       'BPIC_17': 34}

#   outcome_bool 
outcome_bools_dict = {'BPIC_19': False,
                      'BPIC_17_DR': True,
                      'BPIC_17': True}

#   out_mask
out_masks_dict = {'BPIC_19' : False, 
                  'BPIC_17_DR' : True, 
                  'BPIC_17' : True}

#   out_type
out_types_dict = {'BPIC_19' : None, 
                  'BPIC_17_DR' : 'multiclass_outcome', 
                  'BPIC_17' : 'multiclass_outcome'}
#   num_outclasses
num_outclasses_dict = {'BPIC_19' : None,
                       'BPIC_17_DR' : 3,
                       'BPIC_17' : 3}

#   outcome_determining_activities (axiom 2)
#   Maps each outcome class index to the activity whose LAST occurrence in the
#   suffix implies that outcome. `None` for logs without an outcome head.
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
outcome_determining_activities_dict = {'BPIC_19' : None,
                                       'BPIC_17_DR' : {0: 'O_Accepted',
                                                       1: 'O_Cancelled',
                                                       2: 'O_Refused'},
                                       'BPIC_17' : {0: 'O_Accepted',
                                                    1: 'O_Cancelled',
                                                    2: 'O_Refused'}}
