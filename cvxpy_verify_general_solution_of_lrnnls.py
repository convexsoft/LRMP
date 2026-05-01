import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import pinv
from pathlib import Path

np.set_printoptions(precision=15)

script_name = Path(__file__).stem


def iff(cond, true_str, false_str):
    return true_str if cond else false_str


def build_connected_laplacian(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    W = np.zeros((n, n))

    # chain edges
    for i in range(n - 1):
        w = 0.5 + rng.random()
        W[i, i + 1] = w
        W[i + 1, i] = w

    # extra edges
    extra_edges = int(round(n))
    for _ in range(extra_edges):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            W[i, j] = W[i, j] + 0.2 * rng.random()
            W[j, i] = W[i, j]

    np.fill_diagonal(W, 0.0)
    d = W.sum(axis=1)
    L = np.diag(d) - W
    L = 0.5 * (L + L.T)
    return W, L


def fix_duals_by_kkt(A, L, x_opt, lam_raw, mu_raw):
    candidates = [
        ("(lam, mu)", lam_raw,  mu_raw),
        ("(-lam,-mu)", -lam_raw, -mu_raw),
        ("(-lam, mu)", -lam_raw,  mu_raw),
        ("( lam,-mu)",  lam_raw, -mu_raw),
    ]

    best = None
    for name, lam, mu in candidates:
        res = np.linalg.norm(L @ x_opt + A.T @ lam - mu)
        neg_mu = np.minimum(mu, 0.0)
        penalty = 1e2 * np.linalg.norm(neg_mu)
        score = res + penalty
        if best is None or score < best["score"]:
            best = {"name": name, "lam": lam, "mu": mu, "res": res, "score": score,
                    "min_mu": float(np.min(mu))}

    return best["lam"], best["mu"], best


# -----------------------------
# Problem setup
# -----------------------------
print("=== Problem Setup ===")
seed = 0
rng = np.random.default_rng(seed)

m = 50
n = 30
A = rng.standard_normal((m, n))
b = rng.random(m) + 0.5

W, L = build_connected_laplacian(n=n, seed=seed)

print(f"Generated data: m={m}, n={n}")
print(f"A shape: {A.shape}, b shape: {b.shape}, L shape: {L.shape}\n")


# -----------------------------
# Verify Laplacian Properties
# -----------------------------
print("=== Laplacian Properties ===")
print(f"Symmetry ||L-L.T||_F = {np.linalg.norm(L - L.T, 'fro'):.3e}")
e = np.linalg.eigvalsh(L)
print(f"Min eigenvalue  = {np.min(e):.3e} (should be ~0)")

rank_L = np.linalg.matrix_rank(L)
print(f"Rank(L)         = {rank_L} (should be n-1={n-1} for connected graph)")
print(f"L*1             = {np.linalg.norm(L @ np.ones(n)):.3e} (should be ~0)\n")


# -----------------------------
# Solve LR-NNLS in CVXPY (MOSEK preferred)
#   minimize 0.5||y||^2 + 0.5 x^T L x
#   s.t.      A x - b - y = 0,  x >= 0
# -----------------------------
print("=== Solving LR-NNLS via CVXPY ===")
x = cp.Variable(n)
y = cp.Variable(m)

objective = cp.Minimize(0.5 * cp.sum_squares(y) + 0.5 * cp.quad_form(x, L))

eq_constr = (A @ x - b - y == 0)
ineq_constr = (x >= 0)

problem = cp.Problem(objective, [eq_constr, ineq_constr])


solved = False

for solver in ["MOSEK", "OSQP", "SCS"]:
    try:
        if solver == "SCS":
            problem.solve(solver=solver, verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iters=20000)
        else:
            problem.solve(solver=solver, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            print(f"Solved with {solver} | status: {problem.status} | obj: {problem.value:.6e}")
            solved = True
            break
    except Exception as ex:
        print(f"{solver} failed: {ex}")

if not solved:
    raise RuntimeError("All solvers failed.")

x_opt = x.value.reshape(-1)
y_opt = y.value.reshape(-1)

lam_raw = eq_constr.dual_value.reshape(-1)
mu_raw = ineq_constr.dual_value.reshape(-1)


# -----------------------------
# Fix dual signs automatically
# -----------------------------
lambda_opt, mu_opt, best_info = fix_duals_by_kkt(A, L, x_opt, lam_raw, mu_raw)

print("\n=== Dual Sign Alignment (KKT-based) ===")
print(f"Best case: {best_info['name']}")
print(f"Stationarity residual (best): {best_info['res']:.3e}")
print(f"min(mu) (best): {best_info['min_mu']:.3e}\n")


# -----------------------------
# Verify KKT conditions (with corrected duals)
# -----------------------------
print("=== KKT Conditions Verification ===")
kkt_stationarity = np.linalg.norm(L @ x_opt + A.T @ lambda_opt - mu_opt)
print(f"Stationarity ||Lx* + A^T lambda* - mu*||  = {kkt_stationarity:.3e}")

comp_slack = abs(mu_opt @ x_opt)
print(f"Complementarity |mu*^T x*|                = {comp_slack:.3e}")

print(f"Nonnegativity min(x*)                     = {np.min(x_opt):.3e}")
print(f"Constraint ||Ax* - b - y*||               = {np.linalg.norm(A @ x_opt - b - y_opt):.3e}\n")


# -----------------------------
# Pseudoinverse reconstruction
# -----------------------------
Ldag = pinv(L)
v = mu_opt - A.T @ lambda_opt
z = Ldag @ v


# -----------------------------
# Method 1: Determine c from active constraints
# -----------------------------
print("=== Method 1: Determine c from Zero Components ===")
tol_x = 1e-8
zero_indices = np.where(x_opt <= tol_x)[0]   # primal-based active set
print(f"Components where x*_i ~ 0: {len(zero_indices)} out of {n}")

if len(zero_indices) > 0:
    c_from_zeros = -z[zero_indices]
    c_consistency = float(np.std(c_from_zeros))
    c_method1 = float(np.mean(c_from_zeros))
    print(f"Consistency check: std(c from zeros) = {c_consistency:.3e}")
    print(f"Determined c = {c_method1:.6e}")
    if c_consistency < 1e-5:
        print("All zero components yield consistent c")
    else:
        print(f"Warning: inconsistency detected (std = {c_consistency:.3e})")
else:
    c_method1 = 0.0
    c_consistency = 0.0
    print("No (near-)zero components, setting c = 0")

x_method1 = z + c_method1 * np.ones(n)
error_m1 = np.linalg.norm(x_opt - x_method1)
rel_error_m1 = error_m1 / max(np.linalg.norm(x_opt), 1e-16)

print(f"Reconstruction error ||x* - (z + c*1)||   = {error_m1:.3e}")
print(f"Relative error                            = {rel_error_m1:.3e}")
print(f"min(x_reconstructed)                      = {np.min(x_method1):.3e}\n")


# -----------------------------
# Method 2: Least squares fit for c
# -----------------------------
print("=== Method 2: Least Squares Fit (for comparison) ===")
c_method2 = float(np.mean(x_opt - z))
print(f"c (from least squares) = {c_method2:.6e}")

x_method2 = z + c_method2 * np.ones(n)
error_m2 = np.linalg.norm(x_opt - x_method2)
rel_error_m2 = error_m2 / max(np.linalg.norm(x_opt), 1e-16)

print(f"Reconstruction error    = {error_m2:.3e}")
print(f"Relative error          = {rel_error_m2:.3e}")

print(f"\nComparison: |c_method1 - c_method2| = {abs(c_method1 - c_method2):.3e}")
print(iff(abs(c_method1 - c_method2) < 1e-6, "Both methods yield the same c\n", "Methods give different c values\n"))


# -----------------------------
# Method 3: Baseline without c
# -----------------------------
print("=== Baseline: Without c (to show necessity) ===")
x_baseline = z
error_baseline = np.linalg.norm(x_opt - x_baseline)
rel_error_baseline = error_baseline / max(np.linalg.norm(x_opt), 1e-16)

print(f"Reconstruction error ||x* - z||  = {error_baseline:.3e}")
print(f"Relative error                   = {rel_error_baseline:.3e}")
print(f"Error amplification factor       = {error_baseline / max(error_m1, 1e-16):.1f}\n")


# -----------------------------
# Summary table
# -----------------------------
print("=" * 80)
print("                         RECONSTRUCTION SUMMARY                                 ")
print("=" * 80)
print(" Method                     | Abs Error    | Rel Error    | Min Value         ")
print("-" * 80)
print(f" Zero Components (Method 1) | {error_m1:.4e} | {rel_error_m1:.4e} | {np.min(x_method1):+.4e}")
print(f" Least Squares (Method 2)   | {error_m2:.4e} | {rel_error_m2:.4e} | {np.min(x_method2):+.4e}")
print(f" Without c (Baseline)       | {error_baseline:.4e} | {rel_error_baseline:.4e} | {np.min(x_baseline):+.4e}")
print("=" * 80 + "\n")


# -----------------------------
# Final verification checklist
# -----------------------------
print("=" * 80)
print("                     THEORETICAL VALIDATION CHECKLIST                           ")
print("=" * 80)
print(f" 1. KKT stationarity satisfied?       {iff(kkt_stationarity < 1e-5, '[YES]', '[NO] ')} (residual = {kkt_stationarity:.3e})")
print(f" 2. Complementary slackness holds?    {iff(comp_slack < 1e-5, '[YES]', '[NO] ')} (|mu^T x| = {comp_slack:.3e})")
if len(zero_indices) > 0:
    print(f" 3. Zero components consistent?       {iff(c_consistency < 1e-5, '[YES]', '[NO] ')} (std(c) = {c_consistency:.3e})")
else:
    print(" 3. Zero components consistent?       [N/A] (no zero components)")
print(f" 4. Method 1 reconstruction accurate? {iff(rel_error_m1 < 1e-5, '[YES]', '[NO] ')} (rel_err = {rel_error_m1:.3e})")
print(f" 5. Method 1 = Method 2?              {iff(abs(c_method1 - c_method2) < 1e-6, '[YES]', '[NO] ')} (diff = {abs(c_method1 - c_method2):.3e})")
print(f" 6. c is necessary?                   {iff(error_baseline > 10*error_m1, '[YES]', '[NO] ')} (baseline {error_baseline/max(error_m1,1e-16):.0f}x worse)")
print("=" * 80)

plt.show()
