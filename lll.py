import numpy as np


# This file implements: https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm

# Postfixes key:
# o     --> orthogonalized
# _i _j --> indexing by i, j, etc.
# Prefixes key:
# Z_   --> integer type
# R_   --> floating point type


def orthog(R_B, R_Bo, R_μ, seq=None):
    """ Operates on R_Bo and R_μ in place to orthogonalize R_B """
    if seq == None: seq = range(R_B.shape[0])
    # orthogonalize the B_i's for i in seq:
    for i in seq:
        R_Bo[i] = R_B[i]
        for j in range(0, i):
            R_μ[i, j] = np.dot(R_B[i], R_Bo[j]) / np.dot(R_Bo[j], R_Bo[j])
            R_Bo[i] -= R_μ[i, j] * R_Bo[j]


def lll(Z_B, δ=0.75, print_progress=False):
    """ Implementation of the LLL lattice basis reduction algorithm. """
    # initial checks
    assert 0.25 < δ < 1.
    n, m = Z_B.shape
    assert n <= m
    # avoid overwriting the Z_B that was passed to us
    Z_B = np.copy(Z_B)
    # initialize R_Bo and R_μ
    R_Bo = np.zeros((n, m))
    R_μ = np.zeros((n, n))
    R_B = Z_B + 0. # float version of B
    k = 1
    k_max = 0
    orthog(R_B, R_Bo, R_μ, [0, 1]) # make sure we're initially orthogonalized up to k
    while True: # (k < n) # implemented as a break statement
        if print_progress and k_max < k:
            k_max = k
            print(k_max)
        for j in range(k-1, -1, -1):
            if abs(R_μ[k, j]) > 0.5:
                Z_B[k] -= round(R_μ[k, j])*Z_B[j]
                R_B[k] = Z_B[k]
                orthog(R_B, R_Bo, R_μ, [k]) # mutates R_Bo, R_μ
        if np.dot(R_Bo[k], R_Bo[k]) > (δ - R_μ[k, k-1]**2)*np.dot(R_Bo[k-1], R_Bo[k-1]):
            k += 1
            if k < n:
                orthog(R_B, R_Bo, R_μ, [k]) # mutates R_Bo, R_μ
            else: break
        else:
            Z_B[[k-1, k]] = Z_B[[k, k-1]] # swap
            R_B[[k-1, k]] = Z_B[[k-1, k]]
            orthog(R_B, R_Bo, R_μ, [k-1, k]) # mutates R_Bo, R_μ
            k = max(1, k - 1)
    return Z_B



