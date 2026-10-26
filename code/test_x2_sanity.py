"""Sanity unit tests for the X2 tangent-space gradient estimator.

These do NOT touch real w^robust. They verify the FD+tangent+CRN plumbing
recovers known analytic gradients. Any failure blocks interpreting x2_diag.csv.
"""
import numpy as np
from exploratory import _tan, grad_h


def test_grad_h_tan():
    c = 2
    u = np.array([0.9, 0.1])
    u_bar = np.array([0.5, 0.5])
    gh = grad_h(u[None, :], u_bar[None, :], n_pool=300)[0]
    ght = _tan(gh)
    assert np.linalg.norm(ght) > 0
    assert ght[0] * ght[1] < 0, "components must have opposite signs"
    print("[test] grad_h_tan:", np.round(ght, 4), "OK")


def _fd_tan(w_fn, u, d=1e-3):
    c = u.shape[0]
    grad = np.zeros(c)
    for k in range(c):
        v = -d / (c - 1) * np.ones(c); v[k] = d
        grad[k] = (w_fn(u + v) - w_fn(u - v)) / (2 * d)
    return _tan(grad)


def test_mock_linear():
    # w(u)=u_1  -> analytic tangent grad = (.5,-.5) for c=2
    u = np.array([0.6, 0.4])
    gw = _fd_tan(lambda z: z[0], u)
    expect = np.array([0.5, -0.5])
    cos = gw @ expect / (np.linalg.norm(gw) * np.linalg.norm(expect))
    print("[test] mock linear grad:", np.round(gw, 4), "cos=", round(float(cos), 4))
    assert abs(cos) > 0.99


def test_mock_constant():
    u = np.array([0.3, 0.7])
    gw = _fd_tan(lambda z: z.sum(), u)
    print("[test] mock constant grad norm:", round(float(np.linalg.norm(gw)), 6))
    assert np.linalg.norm(gw) < 1e-4


def test_mock_cone():
    # w(u) = -||u - u0||_1, u0=(.9,.1); eccentric point u=(.85,.15)
    u0 = np.array([0.9, 0.1])
    u_bar = np.array([0.5, 0.5])
    u = np.array([0.85, 0.15])
    gw = _fd_tan(lambda z: -np.abs(z - u0).sum(), u)
    gh = _tan(grad_h(u[None, :], u_bar[None, :], n_pool=300)[0])
    s = gh @ gw
    print("[test] cone s =", round(float(s), 4), "||gh||=", round(float(np.linalg.norm(gh)), 3))
    assert abs(s) > 1e-3, "cone s must be nonzero"


if __name__ == "__main__":
    test_grad_h_tan()
    test_mock_linear()
    test_mock_constant()
    test_mock_cone()
    print("[sanity] ALL PASSED")
