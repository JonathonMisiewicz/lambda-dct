from enum import Enum

class WeightRule(Enum):
    unitary = 1
    variational = 2
    cumulant = 3

occupied_int = ["i", "j", "k", "l", "m", "n", "p", "o"]
occupied_ext = [i.upper() for i in occupied_int]
virtual_int = ["a", "b", "c", "d", "e", "f", "g", "v"]
virtual_ext = [i.upper() for i in virtual_int]

