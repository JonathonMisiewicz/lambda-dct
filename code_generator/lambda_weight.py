from itertools import combinations
from .classes import Diagram

def compute_lambda_weight(diagram: Diagram) -> float:
    """ Return the weight factor for a cumulant variable theory. """

    total = 1

    # Our first step is to compute the cumulant subdiagrams. An excitation-type cumulant
    # subdiagram is uniquely identified by its completely contracted de-excitation operators.
    # So our first step is to identify all excitation/de-excitation operators that can be
    # completely contracted.

    # Map operator indices to the indices they're completely contracted with.
    excitation = {}
    deexcitation = {}

    # Map operator i to the operators it connects to if i is not connected to 0.
    # Excitation and de-excitation operators are mapped separately.
    for i, operator in enumerate(diagram.operators[1:], start=1):
        operators_contracted_with = set()
        # Populate connections.
        for index in operator.upper + operator.lower:
            connected_to = index.partner
            if connected_to == 0:
                # This operator cannot be completely contracted in a cumulant subdiagram. Continue.
                break
            else:
                operators_contracted_with.add(connected_to)
        # This operator can be completely contracted in a cumulant subdiagram.
        else:
            target_dict = excitation if operator.is_excitation() else deexcitation
            target_dict[i] = operators_contracted_with

    cumulant_subdiagrams = construct_cumulant_subdiagrams(excitation)
    cumulant_subdiagrams.update(construct_cumulant_subdiagrams(deexcitation))

    # Determine which pairs of cumulant subdiagrams can be part of a contractable partition.
    consistent = {}
    for p1, p2 in combinations(cumulant_subdiagrams, 2):
        consistent[frozenset([p1, p2])] = p1.isdisjoint(p2) or p1.issubset(p2) or p1.issuperset(p2)

    # Brute-force check all sets of partition elements and see if they form a contractable partition.
    # This is a "dumb" algorithm: if the nested partition with A, B didn't work, this algorithm will
    # still check A, B, C, even though this combination isn't viable.
    for num_partition_elts in range(1, len(cumulant_subdiagrams) + 1):
        for combo in combinations(cumulant_subdiagrams, num_partition_elts):
            if all(consistent[frozenset([p1, p2])] for p1, p2 in combinations(combo, 2)):
                total += (-1) ** num_partition_elts

    return total

def construct_cumulant_subdiagrams(excitation: dict[int, set[int]]) -> set[frozenset[int]]:
    """
    Construct all cumulant subdiagams of either excitation/de-excitation type.

    excitation:
        Map from operator indices to the indices it's connected to. keys must all be excitation or
        de-excitation type.

    Output
    ------
    Set of frozensets; Each frozenset consists of the operators of a cumulant subdiagram
    """
    cumulant_subdiagrams = set()

    # Brute force algorithm. Take all possible clusters of operators that can individually be
    # completely contracted. Does this produce a connected subdiagram?

    for num_completely_contracted in range(1, len(excitation) + 1):
        for completely_contracted in combinations(excitation, num_completely_contracted):
            root = completely_contracted[0]
            unconnected = set(completely_contracted[1:])
            subdiagram = excitation[root].copy()
            subdiagram.add(root)
            while unconnected:
                new_unconnected = set()
                for elt in unconnected:
                    if not subdiagram.isdisjoint(excitation[elt]):
                        # This element can contribute to the cumulant subdiagram.
                        subdiagram.add(elt)
                        subdiagram.update(excitation[elt])
                    else:
                        # This element can't contribute to the cumulant subdiagram _yet_.
                        new_unconnected.add(elt)
                if unconnected == new_unconnected:
                    # We added no new operators this iteration. This is not connected.
                    break
                else:
                    unconnected = new_unconnected
            else:
                cumulant_subdiagrams.add(frozenset(subdiagram))
    return cumulant_subdiagrams
