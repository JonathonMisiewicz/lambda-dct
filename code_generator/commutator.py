from collections import Counter
from copy import deepcopy
import itertools
from collections.abc import Iterable

from .classes import Operator, Diagram, HalfLine

###
# This file contains an algorithm to compute all terms of the UCC similarity transformed Hamiltonian.
# What the code generator really needs is to compute all terms of the UCC RDMs, which are just the rank 0 terms.
# We solve this harder problem for historical reasons (DOI: 10.1063/5.0036512). This algorithm is the bottleneck
# of the entire program and should be replaced when computing degree six terms.
###

def commutator_with(diagrams: Iterable[Diagram], simplifications: set[str], cluster_ranks: Iterable[int]) -> list[Diagram]:
    """Compute the result of taking a commutator of the diagrams with excitation and de-excitation operators
    of the given cluster ranks, accounting for the available simplifications.
    """
    diagrams = list(diagrams)
    diagrams = add_new_operator(diagrams, simplifications, cluster_ranks)
    diagrams = contract_new_operator(diagrams, simplifications)
    return filter(lambda x: x.operators[-1].contracted(), diagrams)

def add_new_operator(diagrams: Iterable[Diagram], simplifications: set[str], cluster_ranks: Iterable[int]) -> list[Diagram]:
    """ Use `simplifications` to calla  function that will add a new SQ operator to each diagram. """
    func = fc_only_add if "fc_only" in simplifications else brute_force_add
    return func(diagrams, cluster_ranks)

def brute_force_add(diagrams: Iterable[Diagram], cluster_ranks: Iterable[int]) -> list[Diagram]:
    """
    Return a list of diagrams with all possible (de-)excitation operators appended.
    The possible operators are those of the ranks in cluster_ranks.
    """
    operators = [make_excite_rank(rank) for rank in cluster_ranks]
    operators += [make_deexcite_rank(rank) for rank in cluster_ranks]
    new_diagrams = [Diagram(diagram.operators + [operator]) for diagram, operator in itertools.product(diagrams, operators)]
    return new_diagrams

def fc_only_add(diagrams: Iterable[Diagram], cluster_ranks: Iterable[int]) -> list[Diagram]:
    """
    Return a list of diagrams with a the new diagram appended allowing for a complete contraction.
    If no complete contractions are possible, don't add to the list.
    The possible operators are those of the ranks in cluster_ranks.
    """


    new_diagrams = []

    for diagram in diagrams:
        rank = diagram.excitation_rank()
        if rank is None or abs(rank) not in cluster_ranks:
            continue
        elif rank > 0:
            new_term = make_deexcite_rank(rank)
        elif rank < 0:
            new_term = make_excite_rank(-rank)
        new_diagrams.append(Diagram(diagram.operators + [new_term]))
    return new_diagrams

def make_excite_rank(rank: int) -> Operator:
    """Return an excitation operator of the desired rank."""
    return Operator(Counter({HalfLine(False): rank}), Counter({HalfLine(True): rank}))

def make_deexcite_rank(rank:int) -> Operator:
    """Return a de-excitation operator of the desired rank."""
    temp = make_excite_rank(rank)
    return Operator(temp.lower, temp.upper)

def contract_new_operator(diagrams: list[Diagram], simplifications: set[str]) -> list[Diagram]:
    """
    Given a list of diagrams where the last is uncontracted, return all possible diagrams where the last
    operator is contracted with one of the previous diagrams.
    """
    if not diagrams:
        return diagrams
    num_partner_terms = len(diagrams[0]) - 1
    for partner_idx in range(num_partner_terms):
        # Contract the last operator with operator partner_idx in all possible wyas, including none.
        diagrams = itertools.chain(*[partner_contractions(diagram, partner_idx, simplifications) for diagram in diagrams])
    return list(diagrams)

def partner_contractions(diagram: Diagram, term_num: int, simplifications: set[str]) -> list[Diagram]:
    """
    Perform all contractions (including none) of the given diagram involving the term_num operator.
    Apply the simplifications when determining which contractions to perform.

    term_num
        The index of the term that the last diagram will contract with.
    """
    operators = diagram.operators
    new_diagrams = []
    # Generate all possible contractions.
    contractions1 = row_contractions(operators[term_num].upper, operators[-1].lower, term_num, len(diagram) - 1, simplifications)
    contractions2 = row_contractions(operators[term_num].lower, operators[-1].upper, term_num, len(diagram) - 1, simplifications)
    # New for each possibility, create a new diagram to match.
    for (upper_med, lower_last), (lower_med, upper_last) in itertools.product(contractions1, contractions2):
        new_diagram = deepcopy(operators)
        new_diagram[term_num] = Operator(upper_med, lower_med)
        new_diagram[-1] = Operator(upper_last, lower_last)
        new_diagrams.append(Diagram(new_diagram))
    return new_diagrams

def row_contractions(upper_row: Counter, lower_row: Counter, term_num1: int, term_num2: int, simplifications: set[str]) -> list[tuple[Counter, Counter]]:
    """ Given two rows, return the results of all possible contractions between them, subject
    to the given simplifications.

    upper_row, lower_row
        Two counters. Both should be occupied, or both should be virtual.
    term_num1, term_num2
        The indices of the operators the rows come from.
        The contractions are recorded in the counter by the original symbol plus the term number of what
        it contracts with.
    """
    occupation = HalfLine(True) in lower_row
    symbol = HalfLine(occupation)
    contractions_list = list()
    max_contractions = min(upper_row[symbol], lower_row[symbol])
    min_contractions = max_contractions if "fc_only" in simplifications else 0
    for num_contraction in range(min_contractions, max_contractions + 1): 
        new1 = upper_row.copy()
        new2 = lower_row.copy()
        new1[symbol] -= num_contraction
        new2[symbol] -= num_contraction
        new1[HalfLine(occupation, term_num2)] += num_contraction
        new2[HalfLine(occupation, term_num1)] += num_contraction
        # Purge 0 values from the new rows
        new1 += Counter()
        new2 += Counter()
        contractions_list.append((new1, new2))
    return contractions_list

