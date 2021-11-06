from DICE_L.main import partial_trace, full_contract
from DICE_L.construct_tensor import make_amplitudes, tensor_from_diagram
from DICE_L import main
from copy import deepcopy
import pytest
from DICE_L.data import WeightRule
from DICE_L.tensor import Tensor
from DICE_L.tensor_helper import permute_canonicalize_tensor, seek_equivalents
from DICE_L.classes import Amplitude, Diagram, HalfLine, Operator, Symbol
from collections import Counter

def test_create_tensor():
    a1 = Amplitude("IJ", "ab", "t2")
    a2 = Amplitude("KL", "ab", "t2")
    a3 = Amplitude("KL", "ab", "t2", "aa")
    tensor = Tensor([a1, a2], 1, [["I", "J"], ["K", "L"]], set())
    with pytest.raises(AssertionError):
        Tensor([a1, a3], 1, [["I", "J"], ["K", "L"]], set())

def test_partial_trace():
    tensor = Tensor([Amplitude("IJ", "ab", "t2"), Amplitude("KL", "ab", "t2")], 1, [["I", "J"], ["K", "L"]], set())
    result = partial_trace(tensor, True)
    expected = [Tensor([Amplitude("IJ", "ab", "t2"), Amplitude("IL", "ab", "t2")], 1, [["J"], ["L"]], set())]
    assert expected == result
    tensor = Tensor([Amplitude("i", "I", "f_oo"), Amplitude("Ji", "AB", "t2")], 2, [["I", "J"], ["A", "B"]], set([frozenset([frozenset([Symbol("I")]), frozenset([Symbol("J")])])]))
    with pytest.raises(AssertionError):
        partial_trace(tensor, True)

def test_partial_trace_SI():
    tensor = Tensor([Amplitude("IJ", "ab", "t2", "bb"), Amplitude("KL", "ab", "t2", "bb")], 1, [["I", "J"], ["K", "L"]], set())
    result = partial_trace(tensor, True)
    expected = [Tensor([Amplitude("IJ", "ab", "t2", "bb"), Amplitude("IL", "ab", "t2", "bb")], 1, [["J"], ["L"]], set())]
    assert expected == result
    tensor = Tensor([Amplitude("IJ", "ab", "t2", "ab"), Amplitude("KL", "ab", "t2", "ab")], 1, [["I", "J"], ["K", "L"]], set())
    result = partial_trace(tensor, True)
    expected = [Tensor([Amplitude("IJ", "ab", "t2", "ab"), Amplitude("IL", "ab", "t2", "ab")], 1, [["J"], ["L"]], set())]
    expected.append(Tensor([Amplitude("IJ", "ab", "t2", "ab"), Amplitude("KJ", "ab", "t2", "ab")], 1, [["I"], ["K"]], set()))
    for term in expected:
        assert term in result
    for term in result:
        assert term in expected

def test_make_amplitudes():
    diagram = Diagram([
        Operator(Counter({HalfLine(True, 1): 2}),  Counter({HalfLine(True, 2): 2})),
        Operator(Counter({HalfLine(False, 2): 2}),  Counter({HalfLine(True, 0): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 1): 2})),
    ])
    assert make_amplitudes(diagram, set([0])) == [Amplitude("IJ", "KL"), Amplitude("ab", "IJ"), Amplitude("KL", "ab")]

def test_tensor_from_diagram():
    expected = Tensor([Amplitude("IJ", "ab", "t2"), Amplitude("KL", "ab", "t2")], 1/2, [["I", "J"], ["K", "L"]], set())
    diagram = Diagram([
        Operator(Counter({HalfLine(True, 1): 2}),  Counter({HalfLine(True, 2): 2})),
        Operator(Counter({HalfLine(False, 2): 2}),  Counter({HalfLine(True, 0): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 1): 2})),
    ])
    diagram.prefactor *= 2 # Account for the version of this diagram where the two amplitudes flip order.
    assert expected == tensor_from_diagram(diagram, "Connected", WeightRule.unitary)
    assert expected == tensor_from_diagram(diagram, "Connected", WeightRule.variational)
    assert expected == tensor_from_diagram(diagram, "Connected", WeightRule.cumulant)

def test_print_code():
    tensor = Tensor([Amplitude("IJ", "AB", "t2")], 1, [["I", "J"], ["A", "B"]], set())
    tensor.print_code("c")

def test_seek_equivalents():
    input_tensors = [
        Tensor([Amplitude("IJ", "AB", "t2")], 1, [["I", "J"], ["A", "B"]], set()),
        Tensor([Amplitude("IJ", "BA", "t2")], -1, [["I", "J"], ["A", "B"]], set())
    ]
    expected = [Tensor([Amplitude("IJ", "AB", "t2")], 2, [["I", "J"], ["A", "B"]], set())]
    assert seek_equivalents(input_tensors) == expected

    input_tensors = [
        Tensor([Amplitude("IJ", "AB", "g", include_orbspace=True)], 1, [["I", "J"], ["A", "B"]], set())
    ]
    expected = [Tensor([Amplitude("IJ", "AB", "g", include_orbspace=True)], 1, [["I", "J"], ["A", "B"]], set())]
    assert seek_equivalents(input_tensors) == expected

def test_permute():
    expected = Tensor([Amplitude("IJ", "AB", "t2")], -1, [["I", "J"], ["A", "B"]], set())
    permutation = (0,)
    tensor = Tensor([Amplitude("IJ", "BA", "t2")], 1, [["I", "J"], ["A", "B"]], set())
    flip = []
    result = permute_canonicalize_tensor(permutation, tensor, flip)
    assert expected == result

def test_expand_row():
    tensor = Tensor([Amplitude("i", "I", "f_oo"), Amplitude("Ji", "AB", "t2")], 2, [["I", "J"], ["A", "B"]], set([frozenset([frozenset([Symbol("I")]), frozenset([Symbol("J")])])]))
    expected = [
        Tensor([Amplitude("i", "I", "f_oo"), Amplitude("Ji", "AB", "t2")], 2, [["I", "J"], ["A", "B"]], set()),
        Tensor([Amplitude("i", "J", "f_oo"), Amplitude("Ii", "AB", "t2")], -2, [["I", "J"], ["A", "B"]], set())
    ]
    result = main.expand_antisymmetrizer_row(tensor)
    assert expected == result

def test_multiple():
    tensor = Tensor([Amplitude("IJ", "AB", "t2")], -1, [["I", "J"], ["A", "B"]], set())
    assert tensor.is_multiple(tensor)
    with pytest.raises(AssertionError):
        tensor.is_multiple(5)
