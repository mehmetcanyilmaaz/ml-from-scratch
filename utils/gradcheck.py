
import numpy as np
import matplotlib.pyplot as plt


def numeric_grad(f, x, h=1e-6):
    """Centered-difference gradient of scalar f at x. Works for any shape."""
    x = np.asarray(x, dtype=float)          # fix 1: ints can't hold the nudge
    grad = np.zeros_like(x)                 # now float too, since x is float
    for idx in np.ndindex(x.shape):         # fix 5: every index, any shape
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += h
        x_minus[idx] -= h
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def rel_error(a, b, tiny=1e-12):
    """One number: size of the difference relative to the size of the gradients."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), tiny)

def gradcheck(f, x, analytic_grad, h=1e-6, tol=1e-7, verbose=True):
    """Compare numeric vs analytic gradient. Returns (passed, error)."""
    num = numeric_grad(f, x, h)
    err = rel_error(num, analytic_grad)
    passed = err < tol
    if verbose:
        print(f"{'PASS' if passed else 'FAIL'}  rel error {err:.2e}")
        if not passed:                       # fix 3: show where it disagrees
            print("  numeric: ", num)
            print("  analytic:", np.asarray(analytic_grad))
    return passed, err


def h_sweep(f, x, analytic_grad, h_values):
    """Relative error of the numeric gradient at each step size."""
    return np.array([rel_error(numeric_grad(f, x, h), analytic_grad) for h in h_values])


def plot_sweep(h_values, errors, label):
    plt.loglog(h_values, errors, marker="o", label=label)
    plt.xlabel("h")
    plt.ylabel("relative error")
    plt.legend()


H_VALUES = np.logspace(-1, -12, 12)          # 1e-1, 1e-2, ..., 1e-12

# %% sanity checks
# f(x) = x^2 at 3 -> derivative 6
gradcheck(lambda x: np.sum(x ** 2), np.array([3.0]), np.array([6.0]))

# integer input on purpose: should still give [2, 4], not [0, 0]
gradcheck(lambda x: np.sum(x ** 2), np.array([1, 2]), np.array([2.0, 4.0]))
# %%


"""
this checker works on two conditions
1. the function f must be scalar valued, i.e. it returns a single number
2. the analytic gradient must be provided as a numpy array of the same shape as x
"""