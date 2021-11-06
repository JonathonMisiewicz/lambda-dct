from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD2U

def residual(intermed):
    SD2U.residual(intermed)
    param.D3_cumulant_residual_U(intermed)
    param.S3_cumulant_residual_U(intermed)
    param.S3_opdm_residual_U(intermed)
    param.S3_opdm_product_residual_U(intermed)

def cumulant2(i):
    SD2U.cumulant2(i)
    param.D3_cumulant_U(i)
    param.S3_cumulant_U(i)

def RDM1(i):
    SD2U.RDM1(i)
    param.S3_opdm_U(i)

def product(i):
    SD2U.product(i)
    param.S3_opdm_product_U(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
