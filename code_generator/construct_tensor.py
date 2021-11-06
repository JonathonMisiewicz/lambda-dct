from fractions import Fraction
from math import factorial
import itertools
from typing import Callable, Union
from .classes import Amplitude, Diagram, HalfLine, Symbol
from .data import WeightRule
from .helper import find_parity, get_new_symbol
from .lambda_weight import compute_lambda_weight
from .tensor import Tensor

Externals = list[list[Symbol], list[Symbol]]

def tensor_from_diagram(diagram: Diagram, diagram_class: str, weight_rule: WeightRule) -> Tensor:
    perm_weight = diagram.line_automorphisms()
    expanded_operators = set([0])
    amplitudes = make_amplitudes(diagram, expanded_operators)
    sign = determine_sign(amplitudes)
    externals = get_externals(amplitudes)
    antisym_elts = full_antisym(externals)
    asym_weight, antisymmetrizers = reduce_antisym(amplitudes, antisym_elts)

    # Force all amplitudes to be occupied on top.
    for i, amplitude in enumerate(amplitudes):
        if amplitude.lower[0].occupied:
            amplitude.flip()

    if weight_rule == WeightRule.unitary:
        d_weight = Fraction(diagram.prefactor * asym_weight * sign, perm_weight)
    elif weight_rule == WeightRule.variational:
        operator_weight = diagram.operator_automorphisms()
        d_weight = Fraction(asym_weight * sign, perm_weight * operator_weight)
    elif weight_rule == WeightRule.cumulant:
        operator_weight = diagram.operator_automorphisms()
        weight_l = compute_lambda_weight(diagram)
        d_weight = Fraction(asym_weight * sign * weight_l, perm_weight * operator_weight)
    else:
        raise Exception

    for amplitude in amplitudes:
        amplitude.name = f"t{amplitude.rank}"

    return Tensor(amplitudes, d_weight, externals, antisymmetrizers)

def make_amplitudes(diagram: Diagram, bare_indices: set[int]) -> list[Amplitude]:
    """
    Convert a diagram into a list of amplitudes.
    bare_indices are the operator indices that have trivial coefficient tensor, corresponding to the central operator.
    """

    amplitudes = [ [list(), list()] for i in range(len(diagram))]

    claimed_letters = set()

    for op_idx, operator in enumerate(diagram.operators):
        for row_idx, row in enumerate([operator.upper, operator.lower]):
            for halfline, count in row.items():
                partner = halfline.partner
                # Determine the symbol and the partner index.
                if partner < op_idx:
                    # We already added this term. Skip it!
                    continue
                elif {op_idx, partner} & bare_indices:
                    # This is on or contracted to a bare operator. Special index!
                    external = True
                else:
                    # This is an internal index.
                    external = False

                for _ in range(count):
                    symbol_to_add = Symbol("o" if halfline.occupied else "v")
                    symbol_to_add = get_new_symbol(claimed_letters, symbol_to_add, external)
                    claimed_letters.add(symbol_to_add.letter)
                    amplitudes[op_idx][row_idx].append(symbol_to_add)
                    amplitudes[partner][0 if row_idx else 1].append(symbol_to_add)
            # The symbol sort function should treat this now... I just need to work with symbols.
            amplitudes[op_idx][row_idx].sort()

    return [Amplitude(i, j, "") for i, j in amplitudes]

def determine_sign(amplitudes: list[Amplitude]) -> int:
    """Given a diagram, return its sign, accounting for holes and loops."""

    creators = list(itertools.chain(*[amplitude.upper for amplitude in amplitudes]))
    annihilators = list(itertools.chain(*[amplitude.lower for amplitude in amplitudes]))
    sign = 1
    free_creators, free_annihilators = [], []

    def hole_increment(X):
        # Update the sign for occupied lines.
        return -1 if X.occupied else 1

    while creators:
        C, A = creators.pop(0), annihilators.pop(0)
        while True:
            try:
                new_idx = creators.index(A)
            except ValueError:
                # We have a free end.
                while True:
                    try:
                        new_idx = annihilators.index(C)
                    except ValueError:
                        # We found the other free end.
                        free_creators.append(C)
                        free_annihilators.append(A)
                        break
                    sign *= hole_increment(C)
                    C, _ = creators.pop(new_idx), annihilators.pop(new_idx)
                break
            sign *= hole_increment(A)
            _, A = creators.pop(new_idx), annihilators.pop(new_idx)
            if C == A:
                # Loop closed!
                sign *= -1 * hole_increment(A)
                break

    def find_sorted_parity(L):
        # The free terms should all be "distinguished"...
        return find_parity(L, sorted(L))

    sign *= find_sorted_parity(free_creators) * find_sorted_parity(free_annihilators)

    return sign

def get_externals(amplitudes: list[Amplitude]) -> Externals:
    """ Determine all external indices. """
    first_amplitude = amplitudes.pop(0)
    return [first_amplitude.upper, first_amplitude.lower]

def full_antisym(externals: Externals) -> list[tuple[int, set[Symbol]]]:
    """ Determine the indices that should be antisymmetrized.

    Each tuple in output represents a group of antisymmetric indices.
    The int specifies whether this describes a top or bottom row.
    The symbols themselves are in the set.
    """
    antisym_elts = []
    for i, row in enumerate(externals):
        for l in (True, False):
            temp = set(filter(lambda x: x.occupied == l and x.external, row))
            if len(temp) > 1:
                antisym_elts.append((0 if i else 1, temp))
    return antisym_elts

def reduce_antisym(string_list: list[Amplitude], externals: list[tuple[int, set[Symbol]]]) -> tuple[int, set[frozenset[frozenset[Symbol]]]]:
    """ Given a list of antisymmetrizers, convert antisymmetrization over indices that are
        already antisymmetric to constants. We can thus remove them from the list.
        Return the weight from these redundant antisymmetrizers, and the nonredundant ones.

    Input
    -----
    stringlist:
    externals:
        First elt is the row number. Second elt is the antisymmetric occ/vir indices of that row.
    """
    weight = 1
    antisymmetrizers = set()
    for row_num, row in externals:
        asym_row = []
        for operator in string_list:
            amplitude_row = operator.upper if row_num == 0 else operator.lower
            new_found_chars = frozenset(x for x in amplitude_row if x in row)
            if new_found_chars:
                weight *= factorial(len(new_found_chars))
                asym_row.append(new_found_chars)
        if len(asym_row) > 1:
            antisymmetrizers.add(frozenset(asym_row))
    return weight, antisymmetrizers

