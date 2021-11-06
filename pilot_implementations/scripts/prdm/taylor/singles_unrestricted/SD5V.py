from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, SD4V

def residual(intermed):
    SD4V.residual(intermed)
    param.D5_cumulant_residual_V(intermed)
    param.S5_cumulant_residual_SV(intermed)
    param.S5_cumulant_residual_WV(intermed)
    param.S5_opdm_residual_SV(intermed)
    param.S5_opdm_residual_WV(intermed)
    param.S5_opdm_product_residual_SV(intermed)
    param.S5_opdm_product_residual_WV(intermed)

def cumulant2(i):
    SD4V.cumulant2(i)
    param.D5_cumulant_V(i)
    param.S5_cumulant_SV(i)
    param.S5_cumulant_WV(i)

def RDM1(i):
    SD4V.RDM1(i)
    param.S5_opdm_SV(i)
    param.S5_opdm_WV(i)

def product(i):
    SD4V.product(i)
    param.S5_opdm_product_SV(i)
    param.S5_opdm_product_WV(i)

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf_SI(x, [1, 2])
                      )
