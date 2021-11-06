import numpy as np

from pilot_implementations import multilinear as mla

def dct_fock_transformer(f, RDM1):
    evals, evecs = np.linalg.eigh(RDM1)
    denom = mla.full_broadcaster((evals, evals)) - 1
    inter_f = mla.tensor.contra_transform(f, evecs) / denom
    return mla.tensor.contra_transform(inter_f, evecs.T)

