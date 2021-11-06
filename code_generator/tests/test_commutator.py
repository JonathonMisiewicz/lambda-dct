from collections import Counter
from DICE_L.classes import Diagram, Operator, HalfLine
from DICE_L.commutator import commutator_with

def test_commutator_with():
    argument = Diagram([
        Operator(Counter({HalfLine(True): 1}), Counter({HalfLine(True): 1}))
        ])
    test = list(commutator_with([argument], set(), {2}))
    assert list(commutator_with([argument], set(), {2})) == [
        Diagram([
            Operator(Counter({HalfLine(True, 1): 1}), Counter({HalfLine(True): 1})),
            Operator(Counter({HalfLine(False): 2}), Counter({HalfLine(True): 1, HalfLine(True, 0): 1})),
            ]),
        Diagram([
            Operator(Counter({HalfLine(True): 1}), Counter({HalfLine(True, 1): 1})),
            Operator(Counter({HalfLine(True): 1, HalfLine(True, 0): 1}), Counter({HalfLine(False): 2})),
            ])
    ]
