from __future__ import annotations
from .helper import tensor_flip, simple_to_string
from. classes import Amplitude, Spin, Symbol
from fractions import Fraction

class Tensor:

    def __init__(self, amplitudes: list[Amplitude], weight: float, indices: list[list[Symbol], list[Symbol]], asym: set[frozenset[frozenset[Symbol]]]):
        self.amplitudes = amplitudes
        self.weight = weight

        # Three nested lists. The innermost list consists of symbols that are already antisymmetric, without the antisymmetrizer.
        # The next list out groups groups all symbols that are antisymmetric among themselves, because of the antisymmetrizer.
        # The next list out from that groups all externals.
        self.antisymmetrizers = asym

        self.claimed_symbols = dict() # Map letter to actual symbol objects
        for amplitude in amplitudes:
            for symbol in amplitude.upper + amplitude.lower:
                self.claimed_symbols[symbol.letter] = symbol

        indices_already_symbolic = not indices or isinstance(indices[0][0], Symbol)
        if indices_already_symbolic:
            self.external_indices = indices
        else:
            self.external_indices = [[self.claimed_symbols[char] for char in row] for row in indices]
        # Test invariants
        if len(self.external_indices) == 2:
            assert len(self.external_indices[0]) == len(self.external_indices[1])
        else:
            assert len(self.external_indices) == 0

        if all(x.spinorbital for x in amplitudes):
            self.spinorbital = True
        elif all(not x.spinorbital for x in amplitudes):
            self.spinorbital = False
        else:
            raise AssertionError("If one amplitude is in terms of spatial orbitals, all must be.")

    def __str__(self):
        string = str(self.weight) + " " + " | ".join(str(x) for x in self.amplitudes) + " -> "
        string += " ".join(["".join(str(x) for x in row) for row in self.external_indices])
        return string

    def __eq__(self, other):
        return self.is_multiple(other) and self.weight == other.weight

    def rank(self) -> int:
        """ How many external indices are there? """
        return len(self.external_indices[0])

    def is_multiple(self, other: Tensor) -> bool:
        """ Is other a multiple of this tensor? """
        assert isinstance(other, Tensor)
        return self.amplitudes == other.amplitudes and self.external_indices == other.external_indices and self.antisymmetrizers == other.antisymmetrizers

    def spin_suffix(self) -> str:
        """ Return the part of the tensor name due to spin. """
        if not self.external_indices or self.spinorbital:
            return ""
        else:
            return "_" + "".join(str(x.spin) for x in self.external_indices[0])

    def print_code(self, variable: str) -> str:
        """ Print the string representation of the contraction, for use in the pilot implementation. """
        ws = " " * 4
        string = ws + "temp = "
        if self.weight != Fraction(1, 1):
            string += "{} * ".format(self.weight)
        string += "einsum(\""
        flipped_external = tensor_flip(self.external_indices)
        string += ", ".join(i.reduced_str() for i in  self.amplitudes) + " -> " + simple_to_string(flipped_external) + "\", "
        tensor_names = []
        for tensor in self.amplitudes:
            tensor_name = "i[\"" + tensor.full_name() + "\"]"
            tensor_names.append(tensor_name)
        string += ", ".join(tensor_names) + ")\n" + ws + variable + " += "
        if self.antisymmetrizers:
            string += "mla.antisymmetrize_axes_plus(temp"
            group_strings = []
            externals = []
            # Create a list that functions as a hash from symbol to index in einsum
            for row in flipped_external:
                externals += row
            for asym_group in self.antisymmetrizers:
                block_list = []
                for block in asym_group:
                    block_list.append("(" + ", ".join([str(externals.index(symb)) for symb in block]) + ",)")
                string += ", (" + ", ".join(block_list) + ")"
            string+= ")"
        else:
            string += "temp"
        return string

