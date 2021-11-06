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
    # There are no opdm product terms until D4.

def cumulant2(i):
    param.D2_cumulant(i)

def RDM1(i):
    param.D2_opdm(i)

def product(i):
    pass

def intermed(i):
    return build_intermed_SI(cumulant2, RDM1, product, i)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = common.initialize_intermediates_SI
                      )
simultaneous.SI = True
