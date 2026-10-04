import numpy as np
import cvxpy as cp
from scipy.linalg import pinv

np.set_printoptions(precision=15)


def build_connected_laplacian(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    W = np.zeros((n, n))

    # Chain edges guarantee connectivity.
    for i in range(n - 1):
        w = 0.5 + rng.random()
        W[i, i + 1] = w
        W[i + 1, i] = w

    # Add extra random edges.
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


def lr_nnls_objective(A, b, L, x):
    """Evaluate 0.5||Ax-b||^2 + 0.5 x^T L x."""
    r = A @ x - b
    return 0.5 * (r @ r) + 0.5 * (x @ (L @ x))


# ---------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------
print("=== Problem Setup ===")
seed = 0
rng = np.random.default_rng(seed)

m = 50
n = 30

A = rng.standard_normal((m, n))
b = rng.random(m) + 0.5
W, L = build_connected_laplacian(n=n, seed=seed)

one = np.ones(n)

print(f"Generated data: m={m}, n={n}")
print(f"A shape: {A.shape}, b shape: {b.shape}, L shape: {L.shape}\n")


# ---------------------------------------------------------------------
# Verify Laplacian properties
# ---------------------------------------------------------------------
print("=== Laplacian Properties ===")
eig_L = np.linalg.eigvalsh(L)

print(f"Symmetry ||L-L.T||_F = {np.linalg.norm(L - L.T, 'fro'):.3e}")
print(f"Min eigenvalue        = {np.min(eig_L):.3e} (should be ~0)")
print(f"Rank(L)               = {np.linalg.matrix_rank(L)} "
      f"(should be n-1={n-1} for a connected graph)")
print(f"||L1||_2              = {np.linalg.norm(L @ one):.3e} (should be ~0)\n")


# ---------------------------------------------------------------------
# Solve LR-NNLS with CVXPY
#
#   minimize    0.5||y||^2 + 0.5 x^T L x
#   subject to  Ax - b - y = 0
#               x >= 0
#
# CVXPY is used here only to obtain a reference KKT point for
# validating Theorem 1. It is not used in the formula for c^star.
# ---------------------------------------------------------------------
print("=== Solving LR-NNLS via CVXPY ===")

x = cp.Variable(n)
y = cp.Variable(m)

objective = cp.Minimize(
    0.5 * cp.sum_squares(y) + 0.5 * cp.quad_form(x, L)
)

eq_constr = (A @ x - b - y == 0)
ineq_constr = (x >= 0)

problem = cp.Problem(objective, [eq_constr, ineq_constr])

solved = False

for solver in ["MOSEK", "OSQP", "SCS"]:
    try:
        if solver == "SCS":
            problem.solve(
                solver=solver,
                verbose=False,
                eps_abs=1e-9,
                eps_rel=1e-9,
                max_iters=100000,
            )
        elif solver == "OSQP":
            problem.solve(
                solver=solver,
                verbose=False,
                eps_abs=1e-9,
                eps_rel=1e-9,
                max_iter=200000,
                polishing=True,
            )
        else:
            problem.solve(solver=solver, verbose=False)

        if problem.status in ["optimal", "optimal_inaccurate"]:
            print(
                f"Solved with {solver} | status: {problem.status} | "
                f"obj: {problem.value:.12e}"
            )
            solved = True
            break

    except Exception as ex:
        print(f"{solver} failed: {ex}")

if not solved:
    raise RuntimeError("All solvers failed.")

x_opt = np.asarray(x.value).reshape(-1)
y_opt = np.asarray(y.value).reshape(-1)


# lambda is associated with Ax-b-y=0 and mu >= 0 with x >= 0.
lambda_opt = np.asarray(eq_constr.dual_value).reshape(-1)
mu_opt = np.asarray(ineq_constr.dual_value).reshape(-1)


# ---------------------------------------------------------------------
# Verify the complete KKT system
# ---------------------------------------------------------------------
print("\n=== KKT Conditions Verification ===")

res_y_stationarity = np.linalg.norm(y_opt - lambda_opt)
res_stationarity = np.linalg.norm(
    L @ x_opt + A.T @ lambda_opt - mu_opt
)
res_primal_eq = np.linalg.norm(A @ x_opt - b - y_opt)
res_complementarity = abs(mu_opt @ x_opt)

print(f"||y* - lambda*||_2                    = {res_y_stationarity:.3e}")
print(f"||Lx* + A^T lambda* - mu*||_2        = {res_stationarity:.3e}")
print(f"||Ax* - b - y*||_2                   = {res_primal_eq:.3e}")
print(f"|mu*^T x*|                           = {res_complementarity:.3e}")
print(f"min(x*)                               = {np.min(x_opt):.3e}")
print(f"min(mu*)                              = {np.min(mu_opt):.3e}\n")


print("=== Theorem 1: Closed-Form Null-Space Correction ===")

Ldag = pinv(L)
ell = Ldag @ (mu_opt - A.T @ lambda_opt)

a = A @ one
a_norm = np.linalg.norm(a)

c_lower = -np.min(ell)

print(f"||A1||_2                              = {a_norm:.6e}")
print(f"Feasibility lower bound on c          = {c_lower:.12e}")

tol_a = 1e-12

if a_norm > tol_a:
    c_unconstrained = float(
        a @ (b - A @ ell) / (a @ a)
    )
    c_star = max(c_lower, c_unconstrained)

    print("Case                                  : A1 != 0")
    print(f"Unconstrained scalar minimizer c0     = {c_unconstrained:.12e}")
    print(f"Theorem-1 correction c*               = {c_star:.12e}")
    print(
        "Feasibility bound active?             = "
        f"{c_unconstrained < c_lower}"
    )
else:
    # In the shift-invariant case A1=0, every c >= c_lower has
    # the same objective value. We choose the smallest feasible shift
    # as a canonical representative.
    c_unconstrained = np.nan
    c_star = c_lower

    print("Case                                  : A1 = 0")
    print(
        "Objective is invariant along 1; "
        "choosing the smallest feasible c."
    )
    print(f"Chosen feasible correction c*         = {c_star:.12e}")

x_rec = ell + c_star * one

abs_error = np.linalg.norm(x_rec - x_opt)
rel_error = abs_error / max(np.linalg.norm(x_opt), 1e-16)

obj_opt = lr_nnls_objective(A, b, L, x_opt)
obj_rec = lr_nnls_objective(A, b, L, x_rec)
rel_obj_gap = abs(obj_rec - obj_opt) / max(abs(obj_opt), 1e-16)

print("\n=== Reconstruction Accuracy ===")
print(f"||x_rec - x_CVXPY||_2                = {abs_error:.6e}")
print(f"Relative solution error               = {rel_error:.6e}")
print(f"Relative objective gap                = {rel_obj_gap:.6e}")
print(f"min(x_rec)                            = {np.min(x_rec):.6e}")


# ---------------------------------------------------------------------
# Validation-only diagnostic:
# recover c from the reference x* to confirm the theorem numerically.
# This quantity is NOT used to construct x_rec.
# ---------------------------------------------------------------------
c_reference = float(np.mean(x_opt - ell))

print("\n=== Validation-Only Check ===")
print(f"c from reference x* (not used)        = {c_reference:.12e}")
print(f"|c* - c_reference|                    = {abs(c_star - c_reference):.6e}")


# ---------------------------------------------------------------------
# Baseline: omit the null-space correction (c = 0)
# ---------------------------------------------------------------------
x_no_c = ell.copy()

abs_error_no_c = np.linalg.norm(x_no_c - x_opt)
rel_error_no_c = abs_error_no_c / max(np.linalg.norm(x_opt), 1e-16)

obj_no_c = lr_nnls_objective(A, b, L, x_no_c)
rel_obj_gap_no_c = abs(obj_no_c - obj_opt) / max(abs(obj_opt), 1e-16)

print("\n=== Baseline: Omitting the Null-Space Correction (c=0) ===")
print(f"Relative solution error               = {rel_error_no_c:.6e}")
print(f"Relative objective gap                = {rel_obj_gap_no_c:.6e}")
print(f"min(ell)                              = {np.min(x_no_c):.6e}")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n" + "=" * 88)
print("THEOREM-1 VALIDATION SUMMARY")
print("=" * 88)
print(
    f"{'Method':34s} | {'Rel. Sol. Err.':>14s} | "
    f"{'Rel. Obj. Gap':>14s} | {'Min(x)':>12s}"
)
print("-" * 88)
print(
    f"{'Theorem 1 closed-form c*':34s} | "
    f"{rel_error:14.6e} | {rel_obj_gap:14.6e} | {np.min(x_rec):12.4e}"
)
print(
    f"{'No null-space correction (c=0)':34s} | "
    f"{rel_error_no_c:14.6e} | {rel_obj_gap_no_c:14.6e} | {np.min(x_no_c):12.4e}"
)
print("=" * 88)

print("\nChecks:")
print(
    f"  KKT stationarity residual small?    "
    f"{res_stationarity < 1e-5}"
)
print(
    f"  y-stationarity residual small?      "
    f"{res_y_stationarity < 1e-5}"
)
print(
    f"  Reconstruction feasible?            "
    f"{np.min(x_rec) >= -1e-8}"
)
print(
    f"  Reconstruction matches reference?   "
    f"{rel_error < 1e-5}"
)
print(
    f"  Closed-form c matches reference c?  "
    f"{abs(c_star - c_reference) < 1e-5}"
)
