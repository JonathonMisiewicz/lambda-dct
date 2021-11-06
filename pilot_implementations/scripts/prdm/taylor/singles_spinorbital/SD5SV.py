from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD4SV

def residual(intermed):
    SD4SV.residual(intermed)
    param.D5_cumulant_residual_V(intermed)
    param.S5_cumulant_residual_SV(intermed)
    param.S5_opdm_residual_SV(intermed)
    param.S5_opdm_product_residual_SV(intermed)

def cumulant2(i):
    SD4SV.cumulant2(i)
    param.D5_cumulant_V(i)
    param.S5_cumulant_SV(i)

def RDM1(i):
    SD4SV.RDM1(i)
    param.S5_opdm_SV(i)

def product(i):
    SD4SV.product(i)
    param.S5_opdm_product_SV(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
