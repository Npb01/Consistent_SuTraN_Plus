"""Case-level subsetting of a preprocessed split.

Used for cheap partial-data experiments. Instead of dropping random
instances (which would split a case across the keep/drop boundary), a
fraction of whole cases is selected and every tensor is filtered to those
cases. Shared by the equal-weighting and UW train/eval pipelines.
"""
import numpy as np
import torch


def subset_split_by_case(data_tuple, case_ids, subset_fraction, seed):
    """Select a random case-level subset of a split and filter all tensors to it.

    Parameters
    ----------
    data_tuple : tuple[torch.Tensor]
        The split's tensors, all aligned on the instance (row) dimension.
    case_ids : torch.Tensor
        Original case id per instance; same length as each tensor in
        ``data_tuple``.
    subset_fraction : float
        Fraction of unique cases to keep, in (0, 1]. ``>= 1.0`` is a no-op.
    seed : int
        Seed for the case selection, so the subset is reproducible.

    Returns
    -------
    filtered_data_tuple : tuple[torch.Tensor]
    filtered_case_ids : torch.Tensor
    keep_mask : torch.Tensor
        Boolean mask over the original instances, for filtering tensors kept
        outside ``data_tuple`` (e.g. the outcome masks).
    selected_case_ids : torch.Tensor
    """
    if subset_fraction >= 1.0:
        keep_mask = torch.ones_like(case_ids, dtype=torch.bool)
        return data_tuple, case_ids, keep_mask, torch.unique(case_ids)

    unique_case_ids = torch.unique(case_ids).cpu().numpy()
    num_keep = max(1, int(round(len(unique_case_ids) * subset_fraction)))

    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique_case_ids, size=num_keep, replace=False)
    chosen = torch.as_tensor(chosen, dtype=case_ids.dtype, device=case_ids.device)

    keep_mask = torch.isin(case_ids, chosen)
    filtered_data_tuple = tuple(t[keep_mask] for t in data_tuple)
    return filtered_data_tuple, case_ids[keep_mask], keep_mask, chosen
