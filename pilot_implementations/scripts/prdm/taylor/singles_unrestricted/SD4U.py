from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc, SD3U

def residual(intermed):
    SD3U.residual(intermed)
    param.D4_cumulant_residual_U(intermed)
    param.D4_opdm_residual_U(intermed)
    param.D4_opdm_product_residual(intermed)
    param.S4_cumulant_residual_U(intermed)
    param.S4_opdm_residual_U(intermed)
    param.S4_opdm_product_residual_U(intermed)

def cumulant2(i):
    SD3U.cumulant2(i)
    param.D4_cumulant_U(i)
    param.S4_cumulant_U(i)

def RDM1(i):
    SD3U.RDM1(i)
    param.D4_opdm_U(i)
    param.S4_opdm_U(i)

def product(i):
    SD3U.product(i)
    param.D4_opdm_product(i)
    param.S4_opdm_product_U(i)

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf_SI(x, [1, 2])
                      )
