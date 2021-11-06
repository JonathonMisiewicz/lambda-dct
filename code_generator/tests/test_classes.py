from DICE_L.classes import Amplitude, Diagram, Operator, Spin, Symbol, HalfLine
from collections import Counter
from pynauty import Graph

def test_spin_amplitude():
    amplitude = Amplitude("ab", "KL", "t2", "ab")
    assert amplitude == Amplitude("ab", "KL", "t2", [Spin.ALPHA, Spin.BETA])

def test_symbol():
    assert Symbol("I", Spin.BETA) < Symbol("A", Spin.BETA)

def test_excitation():
    assert Operator(Counter({HalfLine(True, 0): 1, HalfLine(True): 2}), Counter({HalfLine(False, 1): 2, HalfLine(False): 1})).is_deexcitation()
    assert not Operator(Counter({HalfLine(False, 0): 1, HalfLine(False): 2}), Counter({HalfLine(True, 1): 2, HalfLine(True): 1})).is_deexcitation()
    assert not Operator(Counter({HalfLine(False, 0): 1, HalfLine(True): 2}), Counter({HalfLine(False, 1): 2, HalfLine(False): 1})).is_deexcitation()

    assert Operator(Counter({HalfLine(False, 0): 1, HalfLine(False): 2}), Counter({HalfLine(True, 1): 2, HalfLine(True): 1})).is_excitation()
    assert not Operator(Counter({HalfLine(True, 0): 1, HalfLine(True): 2}), Counter({HalfLine(False, 1): 2, HalfLine(False): 1})).is_excitation()
    assert not Operator(Counter({HalfLine(True, 0): 1, HalfLine(False): 2}), Counter({HalfLine(True, 1): 2, HalfLine(True): 1})).is_excitation()

def test_line_weight():
    diagram = Diagram([
        Operator(Counter({HalfLine(True, 1): 2}),  Counter({HalfLine(True, 2): 2})),
        Operator(Counter({HalfLine(False, 2): 2}),  Counter({HalfLine(True, 0): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 1): 2})),
    ])

    assert diagram.line_automorphisms() == 8

def test_nauty():
    diagram = Diagram([
        Operator(Counter({HalfLine(True, 1): 2}),  Counter({HalfLine(True, 2): 2})),
        Operator(Counter({HalfLine(False, 2): 2}),  Counter({HalfLine(True, 0): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 1): 2})),
    ])

    assert diagram.graph() == Graph(number_of_vertices=6, directed=True,
     adjacency_dict = {
      0: [3],
      1: [4],
      2: [5],
      3: [1],
      4: [2],
      5: [0],
     },
     vertex_coloring = [
      set([0]),
      set([1]),
      set([2]),
      set([3, 4, 5]),
     ],
    )

def test_diagram_equiv():
    d1 = Diagram([
        Operator(Counter({HalfLine(True, 1): 2}),  Counter({HalfLine(True, 2): 2})),
        Operator(Counter({HalfLine(False, 2): 2}),  Counter({HalfLine(True, 0): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 1): 2})),
    ])
    d2 = Diagram([
        Operator(Counter({HalfLine(True, 2): 2}),  Counter({HalfLine(True, 1): 2})),
        Operator(Counter({HalfLine(True, 0): 2}),  Counter({HalfLine(False, 2): 2})),
        Operator(Counter({HalfLine(False, 1): 2}),  Counter({HalfLine(True, 0): 2})),
    ])
    assert d1.equiv(d2)
