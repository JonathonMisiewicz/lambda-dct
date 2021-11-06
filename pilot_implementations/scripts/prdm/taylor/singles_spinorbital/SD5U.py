from functools import partial
from pilot_implementations.scripts.prdm import common
from pilot_implementations.scripts.prdm import spinorbital_param as param
from pilot_implementations.scripts.prdm.taylor.proc import build_intermed
from . import proc, SD4U

def residual(intermed):
    SD4U.residual(intermed)
    param.D5_cumulant_residual_U(intermed)
    param.S5_cumulant_residual_U(intermed)
    param.S5_opdm_residual_U(intermed)
    param.S5_opdm_product_residual_U(intermed)
    # There are no opdm product terms until D4.

def cumulant2(i):
    SD4U.cumulant2(i)
    param.D5_cumulant_U(i)
    param.S5_cumulant_U(i)

def RDM1(i):
    SD4U.RDM1(i)
    param.S5_opdm_U(i)

def product(i):
    SD4U.product(i)
    param.S5_opdm_product_U(i)

def intermed(i):
    return build_intermed(cumulant2, RDM1, product, i, True)

simultaneous = partial(proc.simultaneous,
                       compute_intermediates = intermed,
                       compute_amplitude_residual = residual,
                       initialize_intermediates = lambda x: common.initialize_intermediates_hf(x, [1, 2])
                      ) 
