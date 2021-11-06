from pilot_implementations.multilinear.tensor import einsum

def perturbation_gradient(gen_fock, hx, gx, sx, nx, RDM1, RDM2):
    one_term = einsum("q p,p q", hx, RDM1)
    two_term = 1 / 4 * einsum("rs pq,pq rs", gx, RDM2)
    lag_term = - einsum("q p,p q", gen_fock, sx)
    return one_term + two_term + lag_term + nx

