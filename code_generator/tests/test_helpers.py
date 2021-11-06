from DICE_L.main import compute_central_weight
from DICE_L.construct_tensor import get_externals, full_antisym
from DICE_L.helper import get_space_count, get_space_count_spinfree, get_space_count_spinaware, get_space_string, swap_symbols
from DICE_L.classes import Amplitude, Symbol, Spin

import pytest

@pytest.mark.parametrize("inp,output", [
    ({Symbol("A"), Symbol("B")}, {Symbol("d"), Symbol("e")}),
    ([Symbol("A"), Symbol("B")], [Symbol("d"), Symbol("e")]),
    (frozenset([Symbol("A"), Symbol("B")]), frozenset([Symbol("d"), Symbol("e")])),
    (Symbol("A"), Symbol("d")),
    ([[Symbol("A"), Symbol("B")], [Symbol("B"), Symbol("e")]], [[Symbol("d"), Symbol("e")], [Symbol("e"), Symbol("Y")]]),
    ])
def test_swap(inp, output, ids=["set", "list", "frozenset", "char", "nested", "str"]):
    flip_dict = {Symbol("A"): Symbol("d"), Symbol("B"): Symbol("e"), Symbol("e"): Symbol("Y")}
    assert swap_symbols(inp, flip_dict) == output

def test_space_string():
    assert get_space_string([[Symbol("A"), Symbol("I")], [Symbol("J"), Symbol("B")]]) == "voov"

def test_get_external():
    externals = get_externals([Amplitude('IJ', 'AB', ''), Amplitude('AB', 'IJ', '')])
    assert externals == [[Symbol("I"), Symbol("J")], [Symbol("A"), Symbol("B")]]
    antisym = full_antisym(externals)
    assert antisym == [(1, {Symbol("I"), Symbol("J")}), (0, {Symbol("A"), Symbol("B")})]

def test_get_space_count_spinfree():
    symbol_list = [Symbol("I"), Symbol("J"), Symbol("A"), Symbol("B"), Symbol("C")]
    result = get_space_count_spinfree(symbol_list)
    assert result == (2, 3)
    assert result == get_space_count(symbol_list)

def test_get_space_count_spinaware():
    symbol_list = [Symbol("I", Spin.ALPHA), Symbol("J", Spin.BETA), Symbol("A", Spin.BETA), Symbol("B", Spin.BETA), Symbol("C", Spin.ALPHA)]
    result = get_space_count_spinaware(symbol_list)
    assert result == (1, 1, 1, 2)
    assert result == get_space_count(symbol_list)

