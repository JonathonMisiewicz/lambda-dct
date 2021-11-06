from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
import itertools
from .classes import Amplitude, Diagram, Symbol
from .helper import find_parity, get_new_symbol, swap_symbols
from .tensor import Tensor
import sympy

def expand_antisymmetrizers(unexpanded_tensors: Iterable[Tensor]) -> list[Tensor]:
    """
    Return a list of tensors where all antisymmetrized of unexpanded_tensors
    have been expanded.
    """
    fully_expanded_tensors = []
    while unexpanded_tensors:
        tensor = unexpanded_tensors.pop(0)
        expanded_tensors = expand_antisymmetrizer_row(tensor)
        for tensor in expanded_tensors:
            tensor_list = unexpanded_tensors if tensor.antisymmetrizers else fully_expanded_tensors
            tensor_list.append(tensor)
    return fully_expanded_tensors

def expand_antisymmetrizer_row(tensor: Tensor) -> list[Tensor]:
    """
    Given a tensor, expand the first row of antiysmmetrizers into multiple tensors.
    """
    if not tensor.antisymmetrizers:
        return [tensor]
    new_tensors = []
    to_expand_asym = deepcopy(tensor.antisymmetrizers)
    expanding_asym = to_expand_asym.pop()

    # Each element in quotient corresponds to a single i in the mask.
    # Used to generate multiset permutations.
    integer_mask = []
    permutation_blocks = [] # Elt. i lists the symbols in quotient i.
    old_ordered_symbols = [] # Gives the "old order" of symbols. Used to find parity of the transformation.
    for i, quotient in enumerate(expanding_asym):
        ordered_quotient = list(quotient) # Any order will do here.
        integer_mask += [i] * len(quotient)
        permutation_blocks.append(ordered_quotient)
        old_ordered_symbols += ordered_quotient

    for permutation in sympy.utilities.iterables.multiset_permutations(integer_mask):
        # The symbol at position i says which antisymmetric block should have an elt. there.
        
        next_index_per_block = [0] * len(permutation_blocks)
        flip_dict = {}
        new_ordering = []
        for block, new_symbol in zip(permutation, old_ordered_symbols):
            next_index = next_index_per_block[block]
            old_symbol = permutation_blocks[block][next_index]
            flip_dict[old_symbol] = new_symbol
            next_index_per_block[block] += 1
        new_ordered_symbols = swap_symbols(old_ordered_symbols, flip_dict)
        amplitudes = [swap_symbols(amplitude, flip_dict) for amplitude in tensor.amplitudes]
        weight = tensor.weight * find_parity(old_ordered_symbols, new_ordered_symbols)
        new_tensors.append(Tensor(amplitudes, weight, deepcopy(tensor.external_indices), to_expand_asym))
    return new_tensors

# TODO: This is horribly, horribly clunky. Is there a better way to do this?
# Pynauty canonicalization sounds promising, but I have no idea how to make that detect parity or handle hermiticity.
def seek_equivalents(tensors: Iterable[Tensor]) -> list[Tensor]:
    """Given a list of tensors, return a list of only the unique tensors. "Duplicate" tensors are added together.
    Resulting tensors are in a canonical form and have nonzero weight. Resulting tensors are in a canonical form."""
    if not tensors:
        return []

    new_tensors = []

    for tensor in tensors:
        # Find ALL permutations that alphabetize the tensor_names.
        name_to_indices = dict()
        flippable = list()
        for i, amplitude in enumerate(tensor.amplitudes):
            name = amplitude.name
            if name not in name_to_indices: name_to_indices[name] = []
            name_to_indices[name].append(i)
            if amplitude.include_orbspace and sum([i.occupied for i in amplitude.upper]) == sum([i.occupied for i in amplitude.lower]):
                flippable.append(i)
        ordered_names = sorted(name_to_indices)
        permutations = [list(itertools.permutations(name_to_indices[name])) for name in ordered_names]
        flips = list(itertools.chain.from_iterable(itertools.combinations(flippable, r) for r in range(len(flippable) + 1)))

        for flip, *permutations in itertools.product(flips, *permutations):
            permutation = list(itertools.chain(*permutations))
            new_tensor = permute_canonicalize_tensor(permutation, tensor, flip)
            for candidate_tensor in new_tensors:
                if new_tensor.is_multiple(candidate_tensor):
                    candidate_tensor.weight += new_tensor.weight
                    break
            else:
                # We didn't find a match. Try another permutation.
                continue
            # We found a match. Stop trying permutations.
            break
        else:
            new_tensors.append(new_tensor)

    nonzero_tensors = list(filter(lambda x: x.weight, new_tensors))
    return nonzero_tensors

def permute_canonicalize_tensor(permutation: tuple[int], tensor: Tensor, flip: list[int]) -> Tensor:
    """Given a Tensor, permute the amplitudes in it. In the process, it brings it to a "canonical"
    form, to assist in recognizing identical tensors.

    Input
    -----
    permutation:
        A tuple of integers specifying the permutation

    flip: list of int
        Indices (pre-permutation) of top, bottom rows to flip.
    """
    # 1. Construct a map from a future sumbol to a target symbol.
    # The map has the property that equivalent tensors have the same image under applying it.
    # symbol_indices maps a symbol to the indices of the amplitudes that will contain it after the permutation.
    symbol_indices = OrderedDict()
    for future_idx, current_idx in enumerate(permutation):
        amplitude = tensor.amplitudes[current_idx]
        for row in (amplitude.upper, amplitude.lower):
            for symbol in row:
                if symbol not in symbol_indices: symbol_indices[symbol] = []
                symbol_indices[symbol].append(future_idx)
    sorted_symbols = sorted(symbol_indices, key = lambda x: (len(symbol_indices[x]), symbol_indices[x]))
    flip_dict = dict()
    for symbol in sorted_symbols:
        flip_dict[symbol] = get_new_symbol(set(x.letter for x in flip_dict.values()), symbol, len(symbol_indices[symbol]) == 1)

    # 2. Apply flips in flip dict.
    string_list = [(Amplitude(amplitude.lower, amplitude.upper, amplitude.name, include_orbspace=amplitude.include_orbspace) if i in flip else amplitude) for i, amplitude in enumerate(tensor.amplitudes)]

    # 3. Perform substitution of indices in the stringlist, externals, and antisymmetrizers.
    subbed_stringlist = swap_symbols(string_list, flip_dict)
    new_externals = swap_symbols(tensor.external_indices, flip_dict)
    new_antisymmetrizers = swap_symbols(tensor.antisymmetrizers, flip_dict)

    # 4. Next, parity and sort the symbols.
    weight = tensor.weight
    amplitudes = []
    for amplitude in subbed_stringlist:
        new_rows = []
        for i, row in enumerate([amplitude.upper, amplitude.lower]):
            sorted_row = sorted(row)
            weight *= find_parity(row, sorted_row)
            new_rows.append(sorted_row)
        amplitudes.append(Amplitude(*new_rows, amplitude.name, include_orbspace=amplitude.include_orbspace))
    # THIS assumes each row of externals is antisymmetric. That isn't necessarily true if the tensor is expanded....
    for i, row in enumerate(new_externals):
        sorted_row = list(sorted(row))
        weight *= find_parity(row, sorted_row)
        new_externals[i] = sorted_row

    # 5. Permute the string list.
    new_amplitudes = [amplitudes[i] for i in permutation]
    return Tensor(new_amplitudes, weight, new_externals, new_antisymmetrizers)

