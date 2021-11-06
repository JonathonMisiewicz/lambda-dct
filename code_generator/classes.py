from __future__ import annotations
from collections import Counter
from enum import Enum
from fractions import Fraction
import functools
from math import factorial
from typing import Union

import pynauty

from .data import occupied_int

class Operator:
    """ Represents a single operator, with unlabeled indices. """

    def __init__(self, upper: Counter[HalfLine], lower: Counter[HalfLine]):
        # self.upper and self.lower are counters. All elements are strings
        # starting with o and v.
        self.upper = upper
        self.lower = lower

    def __str__(self):
        return " ".join(string_helper(row) for row in [self.upper, self.lower])

    def __eq__(self, other):
        return self.upper == other.upper and self.lower == other.lower

    def rank(self) -> int:
        return sum(self.upper.values())
        # Should always equal sum(self.lower.values())

    def selfadjoint(self) -> bool:
        return self.upper == self.lower

    def contracted(self) -> bool:
        return any(any(halfline.partner is not None for halfline in row) for row in (self.upper, self.lower))

    def is_excitation(self) -> bool:
        return all(not x.occupied for x in self.upper) and all(x.occupied for x in self.lower)

    def is_deexcitation(self) -> bool:
        return all(x.occupied for x in self.upper) and all(not x.occupied for x in self.lower)

    def uncontracteds(self) -> tuple[int, int, int, int]:
        return self.upper[HalfLine(True)], self.upper[HalfLine(False)], self.lower[HalfLine(True)], self.lower[HalfLine(False)]

def string_helper(row: Counter[HalfLine]) -> str:
    return "".join([str(halfline) * mult for halfline, mult in row.items()])

class Diagram:
    """ Represents a diagram with unlabeled indices. """

    def __init__(self, operators: list[Operator]):
        self.operators = operators
        self.prefactor = Fraction(1, factorial(len(operators) - 1))
    
    def __len__(self):
        return len(self.operators)

    def __eq__(self, other):
        return self.operators == other.operators

    def __str__(self):
        return str(self.prefactor) + " * " + " ".join(str(i) for i in self.operators)

    def graph(self):
        num_vertices = len(self.operators)
        adjacency_dict = {x: [] for x in range(len(self.operators))}
        vertex_coloring = [ set([0]), set(),  set()  ]
        # First, mark excitation and de-excitation vertices.
        for i, operator in enumerate(self.operators):
            if not i:
                pass
            elif operator.is_excitation():
                vertex_coloring[1].add(i)
            elif operator.is_deexcitation():
                vertex_coloring[2].add(i)
            else:
                raise Exception("All operators but the first should be particle-hole type.")
            for halfline, count in operator.upper.items():
                destination = halfline.partner
                if destination == None: raise Exception # TODO: Amend?
                if count > 1:
                    adjacency_dict[i].append(num_vertices)
                    adjacency_dict[num_vertices] = [destination]
                    index = count + 1
                    for x in range(index - len(vertex_coloring) + 1):
                        vertex_coloring.append(set())
                    vertex_coloring[index].add(num_vertices)
                    num_vertices += 1
                    pass
                elif count == 1:
                    adjacency_dict[i].append(destination)
                else:
                    raise Exception("Count should be >= 1")

        return pynauty.Graph(
                number_of_vertices = num_vertices,
                directed = True,
                adjacency_dict = adjacency_dict,
                vertex_coloring = vertex_coloring
                )

    def rank(self) -> tuple[int, int, int, int]:
        """ Return (upper occupied, upper virtual, lower occupied, lower virtual)
        uncontracted counts. """
        u_o, u_v, l_o, l_v = 0, 0, 0, 0
        for operator in self.operators:
            a, b, c, d = operator.uncontracteds()
            u_o += a
            u_v += b
            l_o += c
            l_v += d
        return (u_o, u_v, l_o, l_v)

    def ph_symmetric_externals(self) -> bool:
        """ Return whether the uncontracted lines have particle-hole symmetry. """
        u_o, u_v, l_o, l_v = self.rank()
        return u_o == l_o and u_v == l_v

    def excitation_rank(self) -> Union[None, int]:
        """ Return the excitation rank.
        Deexcitation operators have a negative rank. Non-particle-hole operators have None."""
        u_o, u_v, l_o, l_v = self.rank()
        if not u_o and not l_v and u_v == l_o:
            return u_v
        if not u_v and not l_o and u_o == l_v:
            return -u_o
        return None

    def equiv(self, other) -> bool:
        """ Return whether the other diagram is this one, upon some permutation of operators. """
        return pynauty.isomorphic(self.graph(), other.graph())

    def is_tensorially_connected(self) -> bool:
        """ Return whether the diagram is a connected tensor. The final contraction pattern is connected.
        If it wasn't, we wouldn't have a complete contraction. But because the central RDM operator has
        a coefficient of 1, that doesn't guarantee that the final tensor is connected. Connected contractions
        meaning a connected term only holds if all your coefficients are size-extensive. """
        return self.connectivity_helper({0, 1}, [1])

    def connectivity_helper(self, bar_path_through: set[int], connected_starters: Iterable[int]) -> bool:
        """ Return whether all operators (minus those in bar_path_through) are connected to the operators
        in connected_starters by a path not through bar_path_through."""
        unprocessed_indices = list(connected_starters)
        connected_indices = set(connected_starters).union(bar_path_through)
        while unprocessed_indices:
            operator = self.operators[unprocessed_indices.pop(0)]
            for row in [operator.upper, operator.lower]:
                for key in row:
                    connected_index = key.partner
                    if connected_index and connected_index not in connected_indices:
                        connected_indices.add(connected_index)
                        unprocessed_indices.append(connected_index)
            if len(connected_indices) == len(self):
                return True
        return False

    def is_connected_without(self, reject: int) -> bool:
        """ Return whether the diagram would be connected without operator `reject`."""
        return self.connectivity_helper({0, reject}, [0])

    def is_strongly_connected(self) -> bool:
        return all(self.is_connected_without(x) for x in range(1, len(self)))

    def line_automorphisms(self) -> int:
        """ Get the permutational weight."""
        # "labeled" lines are not treated specially and are handled elsewhere
        product = 1
        for i, operator in enumerate(self.operators):
            for row in (operator.upper, operator.lower):
                for key, value in row.items():
                    partner = key.partner
                    # This restriction is to avoid double-counting contracted lines
                    if partner is None or int(partner) > i:
                        product *= factorial(value)
        return product

    def find_class(self) -> str:
        """ Given a diagram with variational stationarity, determine its class."""
        s1 = "Connected" if self.is_tensorially_connected() else "Disconnected"
        s2 = " (Strong)" if self.is_strongly_connected() else " (Weak)"
        return s1 + s2

    def operator_automorphisms(self) -> int:
        autinfo = pynauty.autgrp(self.graph())
        return int(autinfo[1] * 10 ** autinfo[2])


class Spin(Enum):
    NONE = 0
    ALPHA = 1
    BETA = 2

    def __lt__(self, other):
        # https://stackoverflow.com/a/39269589
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __str__(self):
        if self.name == "ALPHA":
            return "α"
        elif self.name == "BETA":
            return "β"
        else:
            return ""

class Symbol():

    def __init__(self, letter: str, spin: Spin = Spin.NONE):
        self.letter = letter
        assert isinstance(spin, Spin)
        self.spin = spin
        self.occupied = letter.lower() in occupied_int
        self.external = letter.isupper()

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.letter == other.letter and self.spin == other.spin

    def __hash__(self):
        return hash((self.letter, self.spin))

    def __str__(self):
        return self.letter + str(self.spin)
    
    def __lt__(self, other):
        """ First, sort by spin. Then, sort by occupation. Then set by capitalization."""
        if self.spin != other.spin:
            return self.spin < other.spin
        elif self.occupied != other.occupied:
            return self.occupied
        elif self.external ^ other.external: # XOR
            return self.external
        else:
            return self.letter < other.letter

def amplitude_parser_helper(inp: Union[str, list[Symbol]], spin: Spin) -> list[Symbol]:
    """ Construct a list of symbols from a string format. Can also return a list of symbols. """
    if isinstance(inp, str):
        if spin is None:
            return [Symbol(letter) for letter in inp]
        else:
            return [Symbol(*x) for x in zip(inp, spin)]
    elif isinstance(inp, list) and all(map(lambda x: isinstance(x, Symbol), inp)):
        # If we have a list of symbols, don't interfere with it.
        return inp
    else:
        raise Exception(f"Unrecognized type {type(inp)}.")

class Amplitude():

    def __init__(self, upper: Union[str, list[Symbol]], lower: Union[str, list[Symbol]], name: str = "", spin: Union[str, None, Iterable[Spin]] = None, include_orbspace: bool = False):
        # spin is used to resolve the spins of symbols, if not given.
        if isinstance(spin, str):
            spin = [Spin.ALPHA if x == "a" else Spin.BETA for x in spin.lower()]
        self.upper = amplitude_parser_helper(upper, spin)
        self.lower = amplitude_parser_helper(lower, spin)
        self.name = name
        assert len(self.upper) == len(self.lower)
        self.rank = len(self.upper)
        self.spin = [symbol.spin for symbol in self.upper]
        assert self.spin == [symbol.spin for symbol in self.lower]
        if set(self.spin) == {Spin.NONE}:
            self.spinorbital = True
        elif Spin.NONE in self.spin:
            raise AssertionError("Amplitudes must either have spatial orbitals or spinorbitals.")
        else:
            self.spinorbital = False

        self.include_orbspace = include_orbspace

    def __eq__(self, other):
        assert isinstance(other, Amplitude)
        return self.upper == other.upper and self.lower == other.lower and self.name == other.name and self.spin == other.spin

    def __str__(self):
        return "".join(str(x) for x in self.upper) + " " + "".join(str(x) for x in self.lower)

    def flip(self):
        self.upper, self.lower = self.lower, self.upper

    def full_name(self) -> str:
        """ Return the full name, including any space/spin suffixes."""
        name = self.name
        if self.include_orbspace: name += self.space_suffix()
        return name + self.spin_suffix()

    def reduced_str(self) -> str:
        """ Return a minimal string representation, used for einsum printing. """
        return "".join(x.letter for x in self.upper) + " " + "".join(x.letter for x in self.lower)

    def space_suffix(self) -> str:
        """ Return a string to specify the external spaces. """
        return "_" + "".join("o" if symbol.occupied else "v" for symbol in (self.upper + self.lower)) 

    def spin_suffix(self) -> str:
        """ Return a string to specify the external spins. """
        if self.upper[0].spin == Spin.NONE:
            return ""
        else:
            return "_" + "".join(str(x.spin) for x in self.upper)

class HalfLine():

    def __init__(self, occupied: bool, partner: Union[None, int] = None):
        self.occupied = occupied
        self.partner = partner

    def __eq__(self, other):
        return self.occupied == other.occupied and self.partner == other.partner

    def __hash__(self):
        return hash((self.occupied, self.partner))

    def __str__(self):
        base = "o" if self.occupied else "v"
        if self.partner is not None:
            base += str(self.partner)
        return base
