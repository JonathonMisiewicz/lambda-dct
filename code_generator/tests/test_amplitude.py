from DICE_L.classes import Amplitude, Symbol, Spin
import pytest

def test_full_name():
    inp = Amplitude("Ii", "Ba", "t2", "ab")
    assert inp.full_name() == "t2_αβ"
    assert inp.spinorbital == False

def test_full_name_spinorbital():
    inp = Amplitude("Ii", "Ba", "t2")
    assert inp.full_name() == "t2"
    assert inp.spinorbital == True

def test_full_name_spinorbital():
    I = Symbol("I", Spin.ALPHA)
    i = Symbol("i")
    B = Symbol("B")
    a = Symbol("a")
    with pytest.raises(AssertionError):
        inp = Amplitude([I, i], [B, a], "t2")

def test_reduced_str():
    inp = Amplitude("Ii", "Ba", "t2", "ab")
    assert inp.reduced_str() == "Ii Ba"
