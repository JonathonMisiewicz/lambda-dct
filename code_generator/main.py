from collections.abc import Iterable
from copy import deepcopy
from fractions import Fraction
import itertools
from math import factorial
from typing import Callable
from .classes import Amplitude, Diagram, Operator, Spin, Symbol
from .tensor import Tensor
from . import commutator
from .spin_integrate import spin_integrate
from .helper import get_space_count, get_space_string, find_parity, multinomial, swap_symbols, write_files
from . import data, helper
from .tensor_helper import expand_antisymmetrizers, expand_antisymmetrizer_row, seek_equivalents
from .construct_tensor import tensor_from_diagram

Stringlist = list[list[list[str], list[str]]]

cluster_ranks = [2]
max_commutator = 4

def full_contract(tensor: Tensor, symbol: str) -> Tensor:
    """ Contract the external indices of a Tensor with a tensor with a new symbol.
        Example use case, going from an RDM formula to the energy contribution.

    Input
    -----
    tensor: Tensor
        The Tensor we're contracting.
    symbol: str
        The string of the thing the Tensor is contracted with. Usually, this is an integral.

    Output
    ------
    Tensor
        The new Tensor
    """
    upper, lower = tensor.external_indices
    new_amplitude = Amplitude(upper, lower, symbol, include_orbspace=True)
    new_amplitudes = [new_amplitude] + tensor.amplitudes
    denominator = factorial(new_amplitude.rank) ** 2 # 1/(n!)^2
    numerator = compute_central_weight(tensor.external_indices) # Prefactor due to choosing a representative ordering: top/bottom/left/right...
    # All antisymmetrized indices are external, and contracted with something else antisymmetric. They simplify to a weight.
    for group in tensor.antisymmetrizers:
        numerator *= multinomial(list(map(len, group)))
    return Tensor(new_amplitudes, tensor.weight * Fraction(numerator, denominator), [], set())

def compute_central_weight(cumulant: list[str, str]) -> int:
    """ Compute the "cumulant weight" associated with the number of ways to produce this cumulant block.
        This accounts for the ways to permute occupied and virtual indices within the top and bottom rows,
        and also the fact that we don't count the hermitian adjoint, if the adjoint is distinct.

    Input
    -----
    cumulant: list of str
        Each list represents a row, top or bottom

    Output
    ------
    int
    """
    top_count = get_space_count(cumulant[0])
    bottom_count = get_space_count(cumulant[1])
    top_weight = multinomial(top_count)
    bot_weight = multinomial(bottom_count)
    # The * 2 accounts for hermitian symmetry
    weight = top_weight * bot_weight * (2 if top_count != bottom_count else 1)
    return int(weight)


def product_rule(tensors: Iterable[Tensor], is_differentiable_name: Callable[str, bool]) -> list[Tensor]:
    """Given a list of Tensors, return the tensors that result upon differentiating with respect to the amplitudes.

    Input
    -----
    tensors: list of Tensor
    is_differentiable_name: function from a tensor name to whether it should be differentiated
    """
    differentiated_tensors = []
    for tensor in tensors:
        assert not tensor.antisymmetrizers
        spin_cases = {Spin.NONE} if tensor.spinorbital else {Spin.ALPHA, Spin.BETA}
        for i in range(len(tensor.amplitudes)):
            if not is_differentiable_name(tensor.amplitudes[i].name): continue
            amplitudes = deepcopy(tensor.amplitudes)
            amplitude = amplitudes.pop(i)
            external = [list(amplitude.upper), list(amplitude.lower)]
            weight = tensor.weight
            antisymmetrizers = deepcopy(tensor.antisymmetrizers) # This shoudl be empty, but...
            # Now let's handle the antisymmetrizers.
            for j, row in enumerate(external):
                for spin_case in spin_cases:
                    asym_chars = frozenset([x for x in row if x.spin == spin_case])
                    asym_row = []
                    for term in amplitudes:
                        found_externals = frozenset(filter(lambda x: x in term.upper or x in term.lower, asym_chars))
                        if found_externals:
                            weight *= factorial(len(found_externals))
                            asym_row.append(found_externals)
                    if len(asym_row) > 1:
                        antisymmetrizers.add(frozenset(asym_row))

            differentiated_tensors.append(Tensor(amplitudes, weight, external, antisymmetrizers))
    return differentiated_tensors

def compute_rdm_param(SQ: Operator, max_commutator: int, cluster_ranks: list[int], weight_rule: data.WeightRule, spinintegrate: bool = False) -> list[dict[str, list[Tensor]]]:
    """
    Input
    -----
    SQ:
        Represents a second quantized operator of generic indices. Each Counter should store only generic
        indices. The first represents the top row, and the second the bottom row.

    Output
    ------
    list of dict of list of Tensor 
        The outer list is the commutator level. Dict is the class.
    """

    # Variable Initialization
    diagram_rank = SQ.rank()
    starting_diagrams = [Diagram([SQ])]
    returns = []

    # All diagrams needed for stationarity conditions. The diagrams you need must be either pure excitation/de-excitation
    # or fully contracted. Variational methods only need fully contracted diagrams. Projective methods only need diagrams
    # that aren't fully contracted, but can fully contract with a single excitation/de-excitation more.
    # By including negative rank operators, we account for excluding the hermitian conjugate of the base operator.
    stationarity_ranks = {0}

    acceptor = False

    for commutator_number in range(1, max_commutator+1):

        # The simplifications are commutator number dependent.
        simplifications = set()
        if commutator_number == max_commutator:
            simplifications.add("fc_only")

        open_diagrams = []
        stationarity_diagrams = {}

        # The expensive step is below.
        new_diagrams = list(commutator.commutator_with(starting_diagrams, simplifications, cluster_ranks))
        for diagram in new_diagrams:
            rank = diagram.excitation_rank()
            # Have we found a diagram that contributes to stationarity conditions?
            if rank in stationarity_ranks:
                diagram_class = diagram.find_class()
                if diagram_class not in stationarity_diagrams: stationarity_diagrams[diagram_class] = []
                stationarity_diagrams[diagram_class].append(diagram)
            # Is the diagram open to further contractions? (Rank not defined or a nonzero integer.)
            if rank is not 0:
                open_diagrams.append(diagram)

        unique_counter = {diagram_class: helper.combine_equivalent_diagrams(diagrams) for diagram_class, diagrams in stationarity_diagrams.items()}
        returns.append({diagram_class: [] for diagram_class in unique_counter})

        for diagram_class, diagrams in unique_counter.items():

            # Uncomment below to filter out doubles only diagrams.
            #diagrams = [x for x in diagrams if any(y.rank() != 2 for y in x.operators[1:])]

            tensors = [tensor_from_diagram(diagram, diagram_class, weight_rule) for diagram in diagrams]
            tensors = [tensor for tensor in tensors if tensor.weight != 0]
            if spinintegrate:
                tensors = list(itertools.chain(*[spin_integrate(tensor) for tensor in tensors]))
                tensors = seek_equivalents(tensors)
            tensors_to_differentiate = []
            tensor_prints = []
            for new_tensor in tensors:
                prefix = "c" if diagram_class.startswith("Connected") and diagram_rank == 2 else "rdm"
                new_ext = helper.tensor_flip(new_tensor.external_indices)
                varname = f"i[\"{prefix}_{get_space_string(new_ext)}{new_tensor.spin_suffix()}\"]"
                tensors_to_differentiate.append(new_tensor)
                tensor_prints.append(new_tensor.print_code(varname))
                returns[-1][diagram_class].append(new_tensor)
            write_files(f"{commutator_number}_{diagram_class.lower()}", "\n".join(tensor_prints))

            # Differentiate.
            if tensors_to_differentiate:
                rdm_to_en_deriv(tensors_to_differentiate, "g" if diagram_rank == 2 else "f", f"{commutator_number}_{diagram_class.lower()}")

        # Use diagrams from n commutators to get those for n+1 commutators
        starting_diagrams = open_diagrams
    return returns

def rdm_to_en_deriv(tensors: Iterable[Tensor], symbol: str, filename: str =""):
    """ Given RDM tensors, print out the tensors for the energy derivatives, assuming a simple product rule.

    tensors: The Tensor objects to differentiate.
    symbol: The symbol of the non-amplitude coefficient. Usually an integral.
    name: name of the file to write to
    """
    tensors = [full_contract(tensor, symbol) for tensor in tensors]
    tensors = product_rule(tensors, lambda x: x.startswith("t"))
    tensors = seek_equivalents(tensors)
    # The code as written spin-integrates the energy and then differentiates that.
    # The code commented below will spin-integrated if not done already. That approach is faster.
    # The speed difference between those two approaches may be significant when the RDM generation
    # is sped up. (See commutator.py.)
    #tensors = list(itertools.chain(*[spin_integrate(tensor) for tensor in tensors]))
    #tensors = seek_equivalents(tensors)
    tensor_prints = [tensor.print_code(f"i[\"r{tensor.rank()}{tensor.spin_suffix()}\"]") for tensor in tensors]
    write_files(f"{filename}_residual", "\n".join(tensor_prints))

def compute_cumulant_partial_trace(data: dict[str, dict[str: list[Tensor]]]):
    """ Compute the d (2-RDM cumulant partial trace) terms of DCT.

    Input
    -----
    data: dict
        A structured dictionary. Key one (e.g., "oo") specifies which block
        is left after partial trace. Key two (e.g., "v") specifies which block
        to partial trace over. The value of that is a list of tensors.
    """
    for d_block, value in data.items():
        for i, (o_data, v_data) in enumerate(zip(value["o"], value["v"]), start=1):
            o_data = compute_d(o_data, True)
            v_data = compute_d(v_data, False)
            d_tensor = seek_equivalents(o_data + v_data)
            d_tensor = [tensor for tensor in d_tensor if tensor.weight]
            tensor_prints = [tensor.print_code(f"i[\"d_{d_block}{tensor.spin_suffix()}\"]") for tensor in d_tensor]
            write_files(f"{i}_d", "\n".join(tensor_prints))
            rdm_to_en_deriv(d_tensor, "ft", f"{i}_d")

def compute_d(tensors: Iterable[Tensor], target_occupied: bool) -> list[Tensor]:
    """ Partial trace the input tensors over indices with the specified occupation."""
    tensors = expand_antisymmetrizers(tensors[:])
    tensors = list(itertools.chain(*[partial_trace(tensor, target_occupied) for tensor in tensors]))
    return tensors

def partial_trace(tensor: Tensor, target_occupied: bool) -> list[Tensor]:
    """ Partial trace the input tensors over indices with the specified occupation, assuming no antisymmetrizers."""
    if tensor.antisymmetrizers:
        raise AssertionError("You must expand antisymmetrizers before you partial trace.")
    # First, I need to identify whether the tensor is spin-integrated.
    possible_spins = {Spin.NONE} if tensor.spinorbital else {Spin.ALPHA, Spin.BETA}
    results = []
    for possible_spin in possible_spins:
        traced_indices = []
        new_external = []
        weight = tensor.weight
        external_indices = deepcopy(tensor.external_indices) # Don't pop from actual row...
        for row in external_indices:
            for i, char in enumerate(row):
                if char.occupied == target_occupied and char.spin == possible_spin:
                    traced_indices.append(row.pop(i))
                    new_external.append(row)
                    weight *= (-1) ** i
                    break
            else:
                # No match in this row. On to the next spin case.
                break
        else:
            # We had a match in each row!
            flip_dict = {traced_indices[1]: traced_indices[0]}
            amplitudes = [swap_symbols(amplitude, flip_dict) for amplitude in tensor.amplitudes]
            results.append(Tensor(amplitudes, weight, new_external, set()))
    return results

