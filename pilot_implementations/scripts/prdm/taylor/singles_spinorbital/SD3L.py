from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD2L

def residual(intermed):
    SD2L.residual(intermed)
    param.S3_cumulant_residual_L(intermed)
    param.S3_opdm_product_residual_L(intermed)

def cumulant2(i):
    SD2L.cumulant2(i)
    param.S3_cumulant_L(i)

def RDM1(i):
    SD2L.RDM1(i)
    # No OPDM contribution at degree 3, even with singles

def product(i):
    SD2L.product(i)
    param.S3_opdm_product_L(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
