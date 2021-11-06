from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD2V

def residual(intermed):
    SD2V.residual(intermed)
    param.D3_cumulant_residual_V(intermed)
    param.S3_cumulant_residual_SV(intermed)
    param.S3_cumulant_residual_WV(intermed)
    param.S3_opdm_residual_SV(intermed)
    param.S3_opdm_residual_WV(intermed)
    param.S3_opdm_product_residual_SV(intermed)
    param.S3_opdm_product_residual_WV(intermed)

def cumulant2(i):
    SD2V.cumulant2(i)
    param.D3_cumulant_V(i)
    param.S3_cumulant_SV(i)
    param.S3_cumulant_WV(i)

def RDM1(i):
    SD2V.RDM1(i)
    param.S3_opdm_SV(i)
    param.S3_opdm_WV(i)

def product(i):
    SD2V.product(i)
    param.S3_opdm_product_SV(i)
    param.S3_opdm_product_WV(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
