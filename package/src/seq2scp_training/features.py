"""Ordered sequence feature views; null charge never means neutral charge."""
import numpy as np
from .charge_features import extract_features,AA20,DEFAULT_PH_VALUES
COMPOSITION=["fraction_"+aa for aa in AA20]+["log1p_length","unsupported_fraction"]
CHARGE=["log1p_length","unsupported_fraction","f_positive_KR","f_negative_DE","fcr_KRDE","signed_ncpr_KRDE","abs_ncpr_KRDE","f_ionizable_sidechains_KRHDECY","pi_free_chain"]+["q_free_chain_per_residue_ph"+format(ph,".12g").replace(".","p") for ph in DEFAULT_PH_VALUES]
COMPOSITION_CHARGE=list(dict.fromkeys(COMPOSITION+CHARGE))
VIEWS={"composition_length":COMPOSITION,"sequence_charge":CHARGE,"composition_charge":COMPOSITION_CHARGE}

def vector(sequence,view):
    if view not in VIEWS:raise ValueError("unknown feature view")
    row=extract_features(sequence,sequence_view="U",full_chain=True)
    return np.asarray([np.nan if row[k] is None else row[k] for k in VIEWS[view]],dtype=np.float64)
