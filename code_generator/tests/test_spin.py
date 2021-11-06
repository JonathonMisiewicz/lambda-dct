from DICE_L.spin_integrate import construct_external_spin_counts, construct_allowed_spins, spin_integrate_case, spin_integrate, assign_external_spin
import pytest
import unittest
from DICE_L.classes import Amplitude, Spin, Symbol
from DICE_L.tensor import Tensor

def test_unique_spins():
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    #external = [[Symbol("I"), Symbol("A")], [Symbol("J"), Symbol("B")]]
    spin_possibilities = set(construct_external_spin_counts(tensor))
    assert spin_possibilities == set([
        ((1, 0, 1, 0), (1, 0, 1, 0)),
        ((1, 0, 0, 1), (1, 0, 0, 1)),
        ((1, 0, 0, 1), (0, 1, 1, 0)),
        ((0, 1, 1, 0), (0, 1, 1, 0)),
        ((0, 1, 0, 1), (0, 1, 0, 1))
        ])

def test_external_to_spin():
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    external_spins = ((1, 0, 0, 1), (0, 1, 1, 0))
    assert assign_external_spin(tensor, external_spins) == {Symbol("I"): Spin.ALPHA, Symbol("J"): Spin.BETA, Symbol("A"): Spin.BETA, Symbol("B"): Spin.ALPHA}

def test_allowed_spins():
    case = unittest.TestCase()
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    external_spins = {Symbol("I"): Spin.ALPHA, Symbol("J"): Spin.BETA, Symbol("A"): Spin.BETA, Symbol("B"): Spin.ALPHA} 
    expected = [
            {Symbol("I"): Spin.ALPHA, Symbol("J"): Spin.BETA, Symbol("A"): Spin.BETA, Symbol("B"): Spin.ALPHA, Symbol("i"): Spin.BETA, Symbol("a"): Spin.BETA},
            {Symbol("I"): Spin.ALPHA, Symbol("J"): Spin.BETA, Symbol("A"): Spin.BETA, Symbol("B"): Spin.ALPHA, Symbol("i"): Spin.ALPHA, Symbol("a"): Spin.ALPHA}
        ]
    results = list(construct_allowed_spins(tensor, external_spins))
    case.assertCountEqual(results, expected)

def test_spin_integrate_case():
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    allowed_spin = {Symbol("I"): Spin.ALPHA, Symbol("J"): Spin.BETA, Symbol("A"): Spin.BETA, Symbol("B"): Spin.ALPHA, Symbol("i"): Spin.BETA, Symbol("a"): Spin.BETA}
    expected = Tensor([Amplitude("Ii", "Ba", "t2", "ab"), Amplitude("Ji", "Aa", "t2", "bb")], -1, [["I", "A"], ["B", "J"]], set())
    result = spin_integrate_case(tensor, allowed_spin)
    assert result == expected

def test_spin_integrate_no_anti():
    case = unittest.TestCase()
    # Testing the entire mechanism.
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    expected = [
        Tensor([Amplitude("Ii", "Ba", "t2", "aa"), Amplitude("Ji", "Aa", "t2", "aa")], 1, [["I", "A"], ["J", "B"]], set()),
        Tensor([Amplitude("Ii", "Ba", "t2", "aa"), Amplitude("iJ", "aA", "t2", "ab")], -1, [["I", "A"], ["B", "J"]], set()),
        Tensor([Amplitude("iI", "Ba", "t2", "ab"), Amplitude("iJ", "Aa", "t2", "ab")], 1, [["A", "I"], ["B", "J"]], set()),
        Tensor([Amplitude("Ii", "aB", "t2", "ab"), Amplitude("Ji", "aA", "t2", "ab")], 1, [["I", "A"], ["J", "B"]], set()),
        Tensor([Amplitude("Ii", "Ba", "t2", "ab"), Amplitude("Ji", "Aa", "t2", "ab")], 1, [["I", "A"], ["J", "B"]], set()),
        Tensor([Amplitude("iI", "aB", "t2", "ab"), Amplitude("iJ", "aA", "t2", "ab")], 1, [["I", "A"], ["J", "B"]], set()),
        Tensor([Amplitude("Ii", "Ba", "t2", "ab"), Amplitude("Ji", "Aa", "t2", "bb")], -1, [["I", "A"], ["B", "J"]], set()),

        Tensor([Amplitude("Ii", "Ba", "t2", "bb"), Amplitude("Ji", "Aa", "t2", "bb")], 1, [["I", "A"], ["J", "B"]], set())
            ]
    results = list(spin_integrate(tensor))
    for term in results:
        assert term in expected
    for term in expected:
        assert term in results

def test_spin_integrate_anti():
    tensor = Tensor([Amplitude("a", "A", "ft"), Amplitude("IJ", "aB", "t2")], 1, [["I", "J"], ["A", "B"]], set([frozenset([frozenset([Symbol("A")]), frozenset([Symbol("B")])])]))
    expected = [
        Tensor([Amplitude("a", "A", "ft", "aa"), Amplitude("IJ", "Ba", "t2", "aa")], -1, [["I", "J"], ["A", "B"]], set(
            [frozenset([frozenset([Symbol("A", Spin.ALPHA)]), frozenset([Symbol("B", Spin.ALPHA)])])]
            )),
        Tensor([Amplitude("a", "A", "ft", "bb"), Amplitude("IJ", "Ba", "t2", "bb")], -1, [["I", "J"], ["A", "B"]], set(
            [frozenset([frozenset([Symbol("A", Spin.BETA)]), frozenset([Symbol("B", Spin.BETA)])])]
            )),
        Tensor([Amplitude("a", "A", "ft", "aa"), Amplitude("IJ", "aB", "t2", "ab")], 1, [["I", "J"], ["A", "B"]], set()),
        Tensor([Amplitude("a", "B", "ft", "bb"), Amplitude("IJ", "Aa", "t2", "ab")], 1, [["I", "J"], ["A", "B"]], set()),
            ]
    results = list(spin_integrate(tensor))
    for term in results:
        assert term in expected
    for term in expected:
        assert term in results


def test_spin_suffix():
    tensor = Tensor([Amplitude("Ii", "Ba", "t2"), Amplitude("Ji", "Aa", "t2")], 1, [["I", "A"], ["J", "B"]], set())
    assert tensor.spin_suffix() == ""
    tensor = Tensor([Amplitude("Ii", "Ba", "t2", "aa"), Amplitude("Ji", "Aa", "t2", "aa")], 1, [["I", "A"], ["J", "B"]], set())
    assert tensor.spin_suffix() == "_αα"

def test_spin_str():
    assert str(Spin.ALPHA) == "α"
    assert str(Spin.BETA) == "β"
