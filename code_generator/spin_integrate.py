from .helper import find_parity, get_space_count_spinfree, swap_symbols
from .classes import Amplitude, Spin, Symbol
from .tensor import Tensor
import itertools
import sympy
from copy import deepcopy
from typing import Generator, Union
from collections.abc import Iterable

SpinCount = tuple[int, int, int, int]
AsymRow = frozenset[frozenset[Symbol]]

def spin_integrate(tensor: Tensor) -> Generator[Tensor, None, None]:
    for external_spin_count in construct_external_spin_counts(tensor):
        external_symbol_to_spin = assign_external_spin(tensor, external_spin_count)
        for asym_tensor in spin_asym_expand(tensor, external_symbol_to_spin):
            for allowed_spin in construct_allowed_spins(asym_tensor, external_symbol_to_spin):
                yield spin_integrate_case(asym_tensor, allowed_spin)

def construct_external_spin_counts(tensor: Tensor) -> Generator[tuple[SpinCount, SpinCount], None, None]:
    """ Given the external indices of a tensor, construct all ways to split its indices into
    occupied and virtual that are not similar under hermitian conjugation.

    Output
    ------
    (top occupied alpha, top occupied beta, top virtual alpha, top virtual beta),
    (bottom occupied alpha, bottom occupied beta, bottom virtual alpha, bottom virtual beta)
    """
    upper, lower = tensor.external_indices
    # m = top occupied, n = top virtual, o = bottom occupied, p = bottom virtual
    m, n = get_space_count_spinfree(upper)
    o, p = get_space_count_spinfree(lower)
    hermitian_externals = m == o and n == p
    for a in range(tensor.rank() + 1):
        for ma in range(max(0, a - n), min(m, a) + 1):
            # ma >= a - n so n >= a - ma = na.
            mb = m - ma
            na = a - ma
            nb = n - na
            # Do not generate spin-blocks that can be obtained from hermitian conjugation of another block.
            new_cap = ma if hermitian_externals else a
            for oa in range(max(0, a - p), min(o, new_cap) + 1):
                ob = o - oa
                pa = a - oa
                pb = p - pa
                yield ((ma, mb, na, nb), (oa, ob, pa, pb))

def assign_external_spin(tensor: Tensor, external_spins: tuple[SpinCount, SpinCount]) -> dict[Symbol, Spin]:
    """
    Assign each external symbol to a spin.
    We reuse old symbols for convenience of calculating parity factor.
    """
    external_symbol_to_spin = dict()
    for row, (oa, ob, va, vb) in zip(tensor.external_indices, external_spins):
        for x in row[:oa]:
            external_symbol_to_spin[x] = Spin.ALPHA
        for x in row[oa:oa+ob]:
            external_symbol_to_spin[x] = Spin.BETA
        for x in row[oa+ob:oa+ob+va]:
            external_symbol_to_spin[x] = Spin.ALPHA
        for x in row[oa+ob+va:]:
            external_symbol_to_spin[x] = Spin.BETA
    return external_symbol_to_spin


def construct_allowed_spins(tensor: Tensor, external_symbol_to_spin: dict[Symbol, Spin]) -> dict[Symbol, Spin]:
    """
    Augment the external symbols to spin with the spins of all non-external symbols, in all possible ways so that
    each tensor has as many alpha as beta indices. In practice, this means S_z conserving.
    """
    external_symbols = list(itertools.chain(*tensor.external_indices))
    internal_symbols = list(filter(lambda x: x not in external_symbols, tensor.claimed_symbols.values()))
    for internal_spins in itertools.product([Spin.ALPHA, Spin.BETA], repeat=len(internal_symbols)):
        symbol_to_spin = {internal_symbol: internal_spin for internal_symbol, internal_spin in zip(internal_symbols, internal_spins)}
        symbol_to_spin.update(external_symbol_to_spin)
        for amplitude in tensor.amplitudes:
            num_upper_alpha = sum(symbol_to_spin[symbol] == Spin.ALPHA for symbol in amplitude.upper)
            num_lower_alpha = sum(symbol_to_spin[symbol] == Spin.ALPHA for symbol in amplitude.lower)
            if num_upper_alpha != num_lower_alpha:
                # Not S_z conserving.
                break
        else:
            yield symbol_to_spin


def spin_integrate_case(tensor: Tensor, symbol_to_spin: dict[Symbol, Spin]) -> Tensor:
    """ Apply spin to all symbols in the externals and amplitudes, resort, and apply the parity factor. """
    weight = tensor.weight

    def row_helper(row: Iterable[Symbol], weight: float) -> tuple[Iterable[Symbol], weight]:
        """ Apply the spin to all symbols in an iterable, resort, and apply the parity factor. """
        new_row = [Symbol(i.letter, symbol_to_spin[i]) for i in row]
        sorted_row = sorted(new_row)
        weight *= find_parity(new_row, sorted_row)
        return sorted_row, weight

    # Treat externals.
    new_externals = []
    for row in tensor.external_indices:
        row, weight = row_helper(row, weight)
        new_externals.append(row)

    # Treat amplitudes
    new_amplitudes = []
    for i, amplitude in enumerate(tensor.amplitudes):
        new_rows = []
        for row in (amplitude.upper, amplitude.lower):
            row, weight = row_helper(row, weight)
            new_rows.append(row)
        new_amplitudes.append(Amplitude(*new_rows, amplitude.name, include_orbspace=amplitude.include_orbspace))

    return Tensor(new_amplitudes, weight, new_externals, tensor.antisymmetrizers)

def multiset_complement(multiset: list, subset: list) -> list:
    """ Return all elements of multiset that are not in subset. """
    complement = []
    for x in set(multiset):
        complement += [x] * (multiset.count(x) - subset.count(x))
    return complement


def asym_from_list(target_list: Iterable) -> frozenset[frozenset]:
    asym = frozenset({frozenset(block) for block in target_list if block})
    if len(asym) < 2:
        # Example: i/j where i and j are opposite spin
        return None
    else:
        return asym


def expand_row_spin(row: AsymRow, spin_assignments: dict[Symbol, Spin]) -> Generator[tuple[dict[Symbol, Symbol], Union[AsymRow, None], Union[AsymRow, None]], None, None]:
    """
    Given an antisymmetrizer and a spin assignments, resolve the antisymmetrizer into an antisymmetrizer
    over alpha orbitals, and one over beta orbitals in all ways. To understand why different ways exist, suppose
    we have an antisymmetrizer over (i, j, k), but (i, j) are already antisymmetric among themselves, and we have two
    alpha symbols. We can say the two alpha symbols are on the antisymmetric positions, or the antisymmetric positions
    are claimed by one alpha and one beta symbol - we no longer have the antisymmetry.
    """

    ## First, prepare intermediates.
    alpha_symbols = [] # All symbols with alpha spin
    beta_symbols = [] # All symbols with beta spin
    integer_mask = [] # Quotient i appears len(i) times here.
    symbols_per_block = [] # The symbols of each block in some arbitrary order.
    for i, quotient in enumerate(row):
        integer_mask += [i] * len(quotient)
        symbols_per_block.append(list(quotient))
        for symbol in quotient:
            if spin_assignments[symbol] == Spin.ALPHA:
                alpha_symbols.append(symbol)
            elif spin_assignments[symbol] == Spin.BETA:
                beta_symbols.append(symbol)
            else:
                raise Exception("All symbols in the antisymmetrizer row should have a definite spin.")

    # Iterate over all ways to choose which block each symbol goes to.
    for alpha_block_per_symbol in sympy.utilities.iterables.multiset_combinations(integer_mask, len(alpha_symbols)):
        beta_block_per_symbol = multiset_complement(integer_mask, alpha_block_per_symbol)
        next_index_per_block = [0] * len(row)
        flip_dict = {}       

        alpha_symbols_per_block = [[] for block in row] # list of List[Symbol]
        for block, alpha_symbol in zip(alpha_block_per_symbol, alpha_symbols):
            next_index = next_index_per_block[block]
            old_symbol = symbols_per_block[block][next_index]
            flip_dict[old_symbol] = alpha_symbol 
            next_index_per_block[block] += 1
            alpha_symbols_per_block[block].append(Symbol(alpha_symbol.letter, Spin.ALPHA))
        alpha_asym = asym_from_list(alpha_symbols_per_block)
        
        beta_symbols_per_block = [[] for block in row]
        for block, beta_symbol in zip(beta_block_per_symbol, beta_symbols):
            next_index = next_index_per_block[block]
            old_symbol = symbols_per_block[block][next_index]
            flip_dict[old_symbol] = beta_symbol
            next_index_per_block[block] += 1
            beta_symbols_per_block[block].append(Symbol(beta_symbol.letter, Spin.BETA))
        beta_asym = asym_from_list(beta_symbols_per_block)

        yield flip_dict, alpha_asym, beta_asym

def spin_asym_expand(tensor: Tensor, spin_assignments: dict[Symbol, Spin]) -> Generator[Tensor, None, None]:
    """
    Put the external spins in all possible positions, collecting terms related by antisymmetry where possible.
    Yields one tensor for each possibility.

    spin_assignments maps the external indices of tensor to a spin.
    """

    old_ordering = list(itertools.chain(*tensor.external_indices))

    assert set(old_ordering) == set(spin_assignments.keys())

    # For each antisymmetrizer row, give all possible ways to distribute the symbols
    asym_data = [list(expand_row_spin(row, spin_assignments)) for row in tensor.antisymmetrizers]

    for spin_combination in itertools.product(*asym_data):
        # spin_combination is a tuple of tuple(flip_dict, alpha_asym, beta_asym). There's one inner tuple
        # for each outermost row of the antisymmetrizer.
        composite_flip_dict = {k: v for flip_dict, _, _ in spin_combination for k, v in flip_dict.items()}
        weight = tensor.weight * find_parity(old_ordering, swap_symbols(old_ordering, composite_flip_dict))
        amplitudes = [swap_symbols(amplitude, composite_flip_dict) for amplitude in tensor.amplitudes]

        asym = set()
        for _, alpha_asym, beta_asym in spin_combination:
            if alpha_asym is not None:
                asym.add(alpha_asym)
            if beta_asym is not None:
                asym.add(beta_asym)

        yield Tensor(amplitudes, weight, deepcopy(tensor.external_indices), asym)

