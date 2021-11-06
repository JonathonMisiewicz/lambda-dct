Depends on psi4, opt_einsum, and scipy.

For the RDM formulas, see `pilot_implementations/scripts/prdm/spinorbital_param.py`.

Warning! Running all tests takes a long time. Even with small basis sets, tensor contractions aren't cheap.
On the developer's laptop, running all tests requires ~3 hours.

Run `python -m pytest pilot_implementations.test.test_dct` to run DCT tests.
Run `python -m pytest pilot_implementations.test.test_o_taylor` to run tests for orbital optimized Taylor series truncations.
Run `python -m pytest pilot_implementations.test.test_gradient` to run tests for analytic gradients of the above methods.
Run `python -m pytest pilot_implementations.test.test_s_taylor` to run tests for orbital optimized Taylor series truncations with singles.
    Warning! Results of this last test suite are not reported in any published preprint, but one that is being finalized.
