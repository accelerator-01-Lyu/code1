"""
Theorem 2 double-counting counterexample (fully numeric, reproducible).

c=2 clusters, centroids v1=-1, v2=+1 on the line, fuzzifier m=2.
Two unlabeled candidates:
  b : boundary point x_b = 0.05  (near the decision boundary, high entropy)
  o : confident cluster-1 point x_o = -2.00 (deep in cluster 1, low entropy but far from mean)
Pool mean membership is balanced: u_bar = (0.5, 0.5).

We compute the reference robust gain w^rob via the exact 1-D KL-DRO dual
  w^rob = sup_{tau>0} { -tau kappa - tau log E_q[exp(-g/tau)] }
with g(s)=H(u)-H((u+lambda s)/(1+lambda)), q=(1-eps)Dir(alpha0 u)+eps Dir(1_c),
and the exploration bonus h=||u-u_bar||_1.
Then show:
  NoExploration (score = w^rob) selects b.
  Full          (score = w^rob + rho h) selects o once rho > rho*.
"""
import numpy as np
from scipy.stats import dirichlet

rng = np.random.default_rng(20261009)

# ---- fixed hyper-parameters ----
m = 2
lam = 1.0
eps_q = 0.10
alpha0 = 50.0
kappa = 0.50
M = 400_000

# ---- FCM memberships from distances (correct exponent on raw distances: 2/(m-1)) ----
def fcm_membership(x, v):
    d = np.abs(x - v)
    u = np.zeros(2)
    for k in range(2):
        u[k] = 1.0 / np.sum((d[k] / d) ** (2.0 / (m - 1)))
    return u

v = np.array([-1.0, 1.0])
x_b = 0.05
x_o = -2.00
u_b = fcm_membership(x_b, v)
u_o = fcm_membership(x_o, v)
u_bar = np.array([0.5, 0.5])

def H(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))

def score(s, u):
    p = (u + lam * s) / (1.0 + lam)
    return H(u) - H(p)

def reference_samples(u, n):
    # mixture: (1-eps) Dir(alpha0 u) + eps Dir(1_c)
    n1 = int(round((1 - eps_q) * n))
    n2 = n - n1
    s1 = dirichlet.rvs(alpha0 * u, size=n1, random_state=rng)
    s2 = dirichlet.rvs(np.ones(2), size=n2, random_state=rng)
    return np.vstack([s1, s2])

def robust_gain(u, kappa):
    s = reference_samples(u, M)
    g = np.array([score(row, u) for row in s])
    Eg = g.mean()
    Var = g.var()
    # 1-D dual maximization over tau
    taus = np.geomspace(0.05, 50.0, 4000)
    best = -np.inf
    for tau in taus:
        logZ = np.log(np.mean(np.exp(-g / tau)))
        val = -tau * kappa - tau * logZ
        if val > best:
            best = val
    return best, Eg, Var, g

w_b, Eg_b, Var_b, g_b = robust_gain(u_b, kappa)
w_o, Eg_o, Var_o, g_o = robust_gain(u_o, kappa)

h_b = np.sum(np.abs(u_b - u_bar))
h_o = np.sum(np.abs(u_o - u_bar))

print("=== memberships ===")
print(f"u_b = ({u_b[0]:.4f}, {u_b[1]:.4f})  H(u_b)={H(u_b):.4f}")
print(f"u_o = ({u_o[0]:.4f}, {u_o[1]:.4f})  H(u_o)={H(u_o):.4f}")
print(f"distances b: d_b1={abs(x_b-v[0]):.3f} d_b2={abs(x_b-v[1]):.3f}")
print(f"distances o: d_o1={abs(x_o-v[0]):.3f} d_o2={abs(x_o-v[1]):.3f}")
print()
print("=== reference gain (kappa=%.2f) ===" % kappa)
print(f"E_q g_b = {Eg_b:.4f}  Var_q g_b = {Var_b:.4f}")
print(f"E_q g_o = {Eg_o:.4f}  Var_q g_o = {Var_o:.4f}")
print(f"w_b^rob = {w_b:.4f}")
print(f"w_o^rob = {w_o:.4f}")
print()
print("=== exploration bonus ===")
print(f"h_b = ||u_b-u_bar||_1 = {h_b:.4f}")
print(f"h_o = ||u_o-u_bar||_1 = {h_o:.4f}")
print()
print("=== selection ===")
print(f"NoExploration: score b={w_b:.4f} vs o={w_o:.4f} -> selects {'b' if w_b>w_o else 'o'}")
dw = w_b - w_o
dh = h_o - h_b
rho_star = dw / dh
print(f"w_b - w_o = {dw:.4f};  h_o - h_b = {dh:.4f}")
print(f"threshold rho* = (w_b-w_o)/(h_o-h_b) = {rho_star:.4f}")
for rho in [0.0, 0.05, 0.10, 0.15, 0.20]:
    sb = w_b + rho * h_b
    so = w_o + rho * h_o
    print(f"  rho={rho:.2f}: Full score b={sb:.4f} vs o={so:.4f} -> selects {'b' if sb>so else 'o'}")
