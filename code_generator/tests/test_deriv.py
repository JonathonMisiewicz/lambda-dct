from DICE_L.main import full_contract, Tensor, product_rule
import pytest
from DICE_L.classes import Amplitude

def test_full_contract():
    spinorbital = Tensor([Amplitude("IJ", "ab", "t2"), Amplitude("ab", "KL", "t2")], 1, [["I", "J"], ["K", "L"]], set())
    expected = Tensor([Amplitude("IJ", "KL", "g"), Amplitude("IJ", "ab", "t2"), Amplitude("ab", "KL", "t2")], 1/4, [], set())
    result = full_contract(spinorbital, "g")
    assert result == expected

def test_full_contract_integrated():
    spinorbital = Tensor([Amplitude("IJ", "ab", "t2", "ab"), Amplitude("ab", "KL", "t2", "ab")], 1, [["I", "J"], ["K", "L"]], set())
    expected = Tensor([Amplitude("IJ", "KL", "g", "ab"), Amplitude("IJ", "ab", "t2", "ab"), Amplitude("ab", "KL", "t2", "ab")], 1, [], set())
    result = full_contract(spinorbital, "g")
    assert result == expected

def test_product_rule():
    tensors_to_product = [Tensor([Amplitude("IJ", "KL", "g"), Amplitude("IJ", "ab", "t2"), Amplitude("KL", "ab", "t2")], 1, [], set())]
    is_diff = lambda x: x == "t2"
    differentiated_tensors = [
            Tensor([Amplitude("IJ", "KL", "g"), Amplitude("KL", "ab", "t2")], 4, [["I", "J"], ["a", "b"]], set()),
            Tensor([Amplitude("IJ", "KL", "g"), Amplitude("IJ", "ab", "t2")], 4, [["K", "L"], ["a", "b"]], set())
        ]
    result_tensors = product_rule(tensors_to_product, is_diff)
    assert differentiated_tensors == result_tensors

def test_product_rule_integrated():
    tensors_to_product = [Tensor([Amplitude("IJ", "KL", "g", "ab"), Amplitude("IJ", "ab", "t2", "ab"), Amplitude("KL", "ab", "t2", "ab")], 1, [], set())]
    is_diff = lambda x: x == "t2"
    differentiated_tensors = [
            Tensor([Amplitude("IJ", "KL", "g", "ab"), Amplitude("KL", "ab", "t2", "ab")], 1, [["I", "J"], ["a", "b"]], set()),
            Tensor([Amplitude("IJ", "KL", "g", "ab"), Amplitude("IJ", "ab", "t2", "ab")], 1, [["K", "L"], ["a", "b"]], set())
        ]
    result_tensors = product_rule(tensors_to_product, is_diff)
    assert differentiated_tensors == result_tensors

