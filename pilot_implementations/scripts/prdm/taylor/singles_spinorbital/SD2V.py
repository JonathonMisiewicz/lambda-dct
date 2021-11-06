from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc

def residual(intermed):
    intermed["r1"] = 0
    intermed["r2"] = 0
    param.D2_cumulant_residual(intermed)
    param.D2_opdm_residual(intermed)
    param.S2_cumulant_residual(intermed)
    param.S2_opdm_residual_SV(intermed)
    param.S2_opdm_residual_WV(intermed)
    param.S2_opdm_product_residual(intermed)

def cumulant2(i):
    param.D2_cumulant(i)
    param.S2_cumulant(i)

def RDM1(i):
    param.D2_opdm(i)
    param.S2_opdm_SV(i)
    param.S2_opdm_WV(i)

def product(i):
    param.S2_opdm_product(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
