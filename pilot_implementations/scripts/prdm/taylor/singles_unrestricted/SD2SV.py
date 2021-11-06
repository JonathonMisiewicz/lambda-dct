from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import unrestricted_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed_SI
from . import proc

def residual(intermed):
    intermed["r2_αα"] = 0 
    intermed["r2_αβ"] = 0 
    intermed["r2_ββ"] = 0 
    param.D2_cumulant_residual(intermed)
    param.D2_opdm_residual(intermed)
    intermed["r1_α"] = 0
    intermed["r1_β"] = 0
    param.S2_cumulant_residual(intermed)
    param.S2_opdm_residual_SV(intermed)
    param.S2_opdm_product_residual(intermed)
    # There are no opdm product terms until D4.

def cumulant2(i):
    param.D2_cumulant(i)
    param.S2_cumulant(i)

def RDM1(i):
    param.D2_opdm(i)
    param.S2_opdm_SV(i)

def product(i):
    param.S2_opdm_product(i)

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf_SI(x, [1, 2])
                      )
