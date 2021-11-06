from collections import Counter

from . import main, data
from .classes import Diagram, HalfLine, Operator
from .tensor import Tensor


###
# These are the lines to adjust to change the generated terms.
max_degree = 4 # Going above 4 gets slow, rapidly.
param_ranks = [2]
weight_rule = data.WeightRule.unitary
spinintegrate = False
###

def variational(counter: Operator) -> list[list[Tensor]]:
    """ Given the central RDM operator, compute the RDM parameterization and its stationarity conditions. """
    return main.compute_rdm_param(counter, max_degree, param_ranks, weight_rule, spinintegrate=spinintegrate)

def extract_cumulant(rdm_param):
    return [degree_n.get("Connected (Strong)", []) + degree_n.get("Connected (Weak)", []) for degree_n in rdm_param]

o = HalfLine(True)
v = HalfLine(False)

OOOO = Operator(Counter({o: 2}), Counter({o: 2}))
VVVV = Operator(Counter({v: 2}), Counter({v: 2}))
OVOV = Operator(Counter({o: 1, v: 1}), Counter({o: 1, v: 1}))
OOOV = Operator(Counter({o: 2}), Counter({o: 1, v: 1}))
OOVV = Operator(Counter({o: 2}), Counter({v: 2}))
OVVV = Operator(Counter({o: 1, v: 1}), Counter({v: 2}))
OO = Operator(Counter({o: 1}), Counter({o: 1}))
OV = Operator(Counter({o: 1}), Counter({v: 1}))
VV = Operator(Counter({v: 1}), Counter({v: 1}))

results = {str(operator): variational(operator) for operator in [OOVV, OOOO, OVOV, OOOV, OVVV, VVVV, OO, OV, VV]}

d_dict = {
        "oo": {"o": extract_cumulant(results["oo oo"]), "v": extract_cumulant(results["ov ov"])},
        "ov": {"o": extract_cumulant(results["oo ov"]), "v": extract_cumulant(results["ov vv"])},
        "vv": {"o": extract_cumulant(results["ov ov"]), "v": extract_cumulant(results["vv vv"])}
}

main.compute_cumulant_partial_trace(d_dict)

