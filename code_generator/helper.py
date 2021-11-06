from typing import Container, List, Union
from collections.abc import Iterable
from fractions import Fraction
from math import factorial

from .classes import Amplitude, Diagram, Spin, Symbol
from . import data

TensorSymbols = Iterable[Iterable[Symbol], Iterable[Symbol]]
SymbolContainer = Union[Symbol, frozenset["SymbolContainer"], set["SymbolContainer"], list["SymbolContainer"], Amplitude]

occupied_contractable = ["i", "j", "k", "l", "m", "n", "p", "o"]
occupied_contractable_distinguished = [i.upper() for i in occupied_contractable]
virtual_contractable = ["a", "b", "c", "d", "e", "f", "g", "v"]
virtual_contractable_distinguished = [i.upper() for i in virtual_contractable]

def get_space_count(symbol_list: Iterable[Symbol]) -> tuple[int]:
    """
    Return the count of occupied and virtual symbols in their various spin cases.
    """
    spin_case = {x.spin for x in symbol_list}
    if Spin.NONE not in spin_case:
        return get_space_count_spinaware(symbol_list)
    elif all(x == Spin.NONE for x in spin_case):
        return get_space_count_spinfree(symbol_list)
    else:
        raise Exception("Can only get counts for spin-free or spin-only cases.")

def get_space_count_spinaware(symbol_list: Iterable[Symbol]) -> tuple[int, int, int, int]:
    """
    Return the count of occupied alpha, occupied beta, virtual alpha, and virtual beta symbols.
    """
    oa = sum(x.occupied and x.spin == Spin.ALPHA for x in symbol_list)
    ob = sum(x.occupied and x.spin == Spin.BETA for x in symbol_list)
    va = sum(not x.occupied and x.spin == Spin.ALPHA for x in symbol_list)
    vb = sum(not x.occupied and x.spin == Spin.BETA for x in symbol_list)
    return (oa, ob, va, vb)

def get_space_count_spinfree(symbol_list: Iterable[Symbol]) -> tuple[int, int]:
    """
    Return the count of occupied and virtual symbols.
    """
    occ_count = sum(x.occupied for x in symbol_list)
    vir_count = sum(not x.occupied for x in symbol_list)
    return (occ_count, vir_count)

def find_parity(list1: list, list2: list) -> int:
    """
    Return the parity of the permutation between the lists.
    1 is odd, -1 is even.
    Outputs 1 or -1, the parity of the permutation to move one list into the other.
    """
    num_flips = 0
    permutation = [list1.index(i) for i in list2]
    num_elts = len(permutation)
    for i in range(num_elts):
        try:
            index_of_i = permutation.index(i)
        except ValueError:
            raise Exception("These lists are not permutations of each other.")
        num_flips += index_of_i
        permutation.pop(index_of_i)
    return (-1) ** num_flips

def tensor_flip(indices: TensorSymbols) -> TensorSymbols:
    """
    Given a list of two lists of symbols, impose an (incomplete) canonicalization of which
    is on top: whichever has more occupied symbols is on top.
    """
    upper, lower = indices
    lower_count = sum(i.occupied for i in lower)
    upper_count = sum(i.occupied for i in upper)
    return indices[::-1] if lower_count > upper_count else indices

def simple_to_string(simple: TensorSymbols) -> str:
    """
    Convert the TensorSymbols to a string.
    """
    return " ".join(["".join([x.letter for x in row]) for row in simple])

def swap_symbols(groups: SymbolContainer, flip_dict: dict[Symbol, Symbol]) -> SymbolContainer:
    """
    Return a copy of the SymbolContainer where all symbols in flip_dict have been replaced
    by their corresponding values.
    """
    if isinstance(groups, frozenset):
        return frozenset(swap_symbols(x, flip_dict) for x in groups)
    if isinstance(groups, set):
        return set(swap_symbols(x, flip_dict) for x in groups)
    elif isinstance(groups, list):
        return list(swap_symbols(x, flip_dict) for x in groups)
    elif isinstance(groups, Amplitude):
        return Amplitude(swap_symbols(groups.upper, flip_dict), swap_symbols(groups.lower, flip_dict), groups.name, include_orbspace=groups.include_orbspace)
    elif isinstance(groups, Symbol):
        try:
            return flip_dict[groups]
        except KeyError:
            return groups
    else:
        raise Exception(f"Type {type(groups)} not recognized.")

def combine_equivalent_diagrams(diagrams: Iterable[Diagram]) -> list[Diagram]:
    """
    Collapse all equivalences in an iterable of diagrams. A more efficient procedure wouldn't have
    generated multiple equivalent terms, but doing it this way is harder to get wrong.
    """
    unique_diagrams = []
    for diagram in diagrams:
        for unique_diagram in unique_diagrams:
            if unique_diagram.equiv(diagram):
                unique_diagram.prefactor += diagram.prefactor
                break
        else:
            unique_diagrams.append(diagram)
    return unique_diagrams

def get_space_string(operator: [list[Symbol], list[Symbol]]) -> str:
    """Given an operator, return the string that specifies its orbital spaces.
       Used to get the name of the variable for code generation."""
    search_string = operator[0] + operator[1]
    space_string = "".join("o" if char.occupied else "v" for char in search_string)
    return space_string

def write_files(filename: str, contents: str):
    with open(f"generated_code/{filename}.txt", "a") as f:
        f.write(contents + "\n\n")

def multinomial(lst: list[int]) -> int:
    """https://stackoverflow.com/a/46378809"""
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def get_new_symbol(claimed_letters: Container[str], char: Symbol, external: bool) -> Symbol:
    """ Return a symbol of the same occupied/virtual status as char, and the desired external/internal letter.
    Symbol may not be in claimed_letters. """
    if char.occupied:
        letter_list = data.occupied_ext if external else data.occupied_int
    else:
        letter_list = data.virtual_ext if external else data.virtual_int
    for letter in letter_list:
        if letter not in claimed_letters:
            return Symbol(letter, spin=char.spin)
    else:
        raise Exception(f"We're out of letters.")

