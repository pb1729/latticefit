import numpy as np

from lll import lll


def lattice_fit(X:np.ndarray, zeta:float, resolution:int=1000, δ:float=0.9, print_progress:bool=False):
    """ given a dataset X of N vectors that are thought to approximately lie on some lattice,
        (with small integer coefficients in front of the basis vectors),
        we try and fit a set of some basis vectors, ans, which form a basis which the datapoints
        of X approximately lie on top of.
        X: (N, dim) [m]   -- set of data points
        zeta: [1/m]       -- fitting parameter, larger means we care more about the error being very small
        resolution: []    -- how far we subdivide the number 1 in calculation
        δ=0.9: []         -- lattice quality, in range(0.25, 1), decrease for faster results
        ans: (N, dim) [m] -- suggested basis vectors, sorted in order of length """
    N, dim = X.shape
    A = np.zeros((N, N + dim), dtype=int)
    for i in range(N):
        A[i, i] = resolution
        A[i, N:] = np.rint(resolution*zeta*X[i])
    # do the computation:
    ans = lll(A, δ=δ, print_progress=print_progress)
    # extract the found basis vectors
    ans = ans[:, N:]
    # sort output by length for convenience
    veclengths = (-ans**2).sum(-1)
    ans = ans[np.argsort(veclengths)]
    # convert back to floats in original scale
    return ans/(resolution*zeta)

def lattice_match(v, X, zeta:float, resolution:int=1000, δ:float=0.9):
  """ get the decomposition of vector v in terms of the lattice basis X
      v: (dim)    [m]
      X: (N, dim) [m]
      zeta: [1/m] -- smaller value than you used to find the lattice in the first place is helpful here """
  N, dim = X.shape
  A = np.zeros((N + 1, N + 1 + dim), dtype=int)
  for i in range(N):
    A[i, i] = resolution
    A[i, N + 1:] = np.rint(resolution*zeta*X[i])
  A[N, N] = resolution
  A[N, N + 1:] = np.rint(resolution*zeta*v)
  # do the computation, freezing up to N (the original basis) to save on computation
  ans = lll(A, δ=δ, freeze=N)
  x = ans[-1]
  if x[N] > 0:
    x = -x
  coeffs = x[:N]//resolution
  err = ((x[N + 1:]/(resolution*zeta))**2).sum()
  return coeffs, err


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if False:
      print("DEMO 1: fit to 2d lattice")
      print("Grey points show the true underlying lattice. Blue is the slightly noisy data.\nGreen is the origin, yellow are the fitted basis vectors.")
      N = 8
      BG = np.random.randint(-4, 5, size=(1000, 2)) @ np.array([[2., -1.], [1., 1.]])
      plt.scatter(BG[:, 0], BG[:, 1], color="grey")
      X = np.random.randint(-2, 3, size=(N, 2)) @ np.array([[2., -1.], [1., 1.]]) + 0.01*np.random.randn(N, 2)
      plt.scatter(X[:, 0], X[:, 1])
      # now do main testing:
      ans = lattice_fit(X, 4.0)
      plt.scatter(ans[:2, 0], ans[:2, 1])
      plt.scatter([0], [0])
      plt.show()
      input("PRESS ENTER TO CONTINUE")
    # try with a bigger lattice:
    print("\n\nDEMO 2: fit to a high-dimensional lattice")
    N = 50
    dim = 260
    basis_sz = 16
    basis = np.random.randn(basis_sz, dim)
    X = np.random.randint(-1, 2, size=(N, basis_sz)) @ basis + 0.01*np.random.randn(N, dim)
    print("fitting progress (out of %d)" % N)
    ans = lattice_fit(X, 10.0, print_progress=True)
    if False:
      print("displaying plot, make a note of the index where the magnitude drops to 0")
      plt.bar(np.arange(ans.shape[0]), (ans**2).sum(-1))
      plt.xlabel("lattice vector index")
      plt.ylabel("lattice vector magnitude")
      plt.show()
      i_end = int(input("At what index does the magnitude drop to 0? >"))
    else:
      i_end = basis_sz
    print("Basis vectors from fitting:")
    print(ans[:i_end])
    input("PRESS ENTER TO CONTINUE")
    print("\n\nDEMO 3: map a vector onto the lattice from the previous part")
    basis = basis[np.argsort((-basis[:, N + 1:]**2).sum(-1))] # sort basis by length so it hopefully roughly matches up with X in terms of vector order
    coeffs = np.random.randint(-1, 2, size=(1, basis_sz))
    v = coeffs @ basis + 0.01*np.random.randn(1, dim)
    v = v.reshape(dim)
    coeffs_hat, err = lattice_match(v, basis, 1.)
    print("err:", err)
    print("true:", coeffs[0])
    print("pred:", coeffs_hat)



