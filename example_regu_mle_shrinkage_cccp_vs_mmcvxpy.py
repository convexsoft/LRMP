import os
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import cvxpy as cp
import networkx as nx


def _script_stem(default="regularized_tyler_shrinkage_sigma"):
    try:
        return os.path.splitext(os.path.basename(__file__))[0]
    except NameError:
        return default


script_name = _script_stem()
save_dir = f"{script_name}_results"
os.makedirs(save_dir, exist_ok=True)


def make_connected_laplacian(n: int, p: float, rng: np.random.Generator):
    ok = False
    A = None
    while not ok:
        upper = (rng.random((n, n)) < p).astype(float)
        A = np.triu(upper, 1)
        A = A + A.T
        if np.count_nonzero(A) == 0:
            continue
        G = nx.from_numpy_array(A)
        ok = nx.is_connected(G)

    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    return A, L


def pearson_corr(a, b, eps=1e-12):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(a @ b / denom)


def spd_sqrt(A, eps_eig=1e-12):
    A = 0.5 * (A + A.T)
    s, V = np.linalg.eigh(A)
    s = np.maximum(s, eps_eig)
    As = (V * np.sqrt(s)) @ V.T
    return 0.5 * (As + As.T)


def inv_spd(A, eps_eig=1e-12):
    A = 0.5 * (A + A.T)
    s, V = np.linalg.eigh(A)
    s = np.maximum(s, eps_eig)
    Ainv = (V * (1.0 / s)) @ V.T
    return 0.5 * (Ainv + Ainv.T)


def normalize_trace(A, target_trace, eps=1e-12):
    A = 0.5 * (A + A.T)
    return A * (target_trace / (np.trace(A) + eps))


def _fmt_optional_float(x, fmt=".6f"):
    if x is None:
        return "None"
    return format(float(x), fmt)


def regularized_objective_scatter(Sigma, Y, gamma=0.0, epsv=1e-12):
    Y = np.asarray(Y)
    K, d = Y.shape
    Sigma = 0.5 * (Sigma + Sigma.T)

    sign, logdetSigma = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return np.inf

    invSigma = inv_spd(Sigma, eps_eig=epsv)
    tr_inv = float(np.trace(invSigma)) + epsv

    s = 0.0
    for k in range(K):
        y = Y[k, :].reshape(-1, 1)
        q = float((y.T @ invSigma @ y).item()) + epsv
        s += np.log(q)

    return float((1.0 + gamma / d) * logdetSigma + (d / K) * s + gamma * np.log(tr_inv))


def regularized_objective_precision(Theta, Y, gamma=0.0, epsv=1e-12):
    Y = np.asarray(Y)
    K, d = Y.shape
    Theta = 0.5 * (Theta + Theta.T)

    sign, logdetTheta = np.linalg.slogdet(Theta)
    if sign <= 0:
        return np.inf

    s = 0.0
    for k in range(K):
        y = Y[k, :].reshape(-1, 1)
        q = float((y.T @ Theta @ y).item()) + epsv
        s += np.log(q)

    return float(-(1.0 + gamma / d) * logdetTheta + (d / K) * s + gamma * np.log(np.trace(Theta) + epsv))


def plugin_rho(Y, eps=1e-12):
    Y = np.asarray(Y)
    K, d = Y.shape

    Rhat = (d / max(K, 1)) * (Y.T @ Y)
    Rhat = 0.5 * (Rhat + Rhat.T)
    Rhat = normalize_trace(Rhat, d, eps=eps)

    trR2 = float(np.trace(Rhat @ Rhat))

    p = float(d)
    n = float(K)
    num = p * p + (1.0 - 2.0 / p) * trR2
    den = (p * p - n * p - 2.0 * n) + (n + 1.0 + 2.0 * (n - 1.0) / p) * trR2

    if abs(den) < eps:
        rho = 0.0
    else:
        rho = num / den

    rho = float(np.clip(rho, 0.0, 1.0))
    return rho, trR2


def select_rho(
    Y,
    gamma=0.0,
    mode="auto",
    rho_value=1e-3,
    rho_min=0.0,
    rho_max=0.02,
    kgeq_d_value=0.0,
    auto_gain=1.0,
    gamma_shrink_factor=10.0,
    eps=1e-12,
):
    Y = np.asarray(Y)
    K, d = Y.shape

    mode = str(mode).lower().strip()
    valid_modes = {"off", "fixed", "auto"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported rho mode '{mode}'. Choose from {valid_modes}.")

    rho_plugin, trR2 = plugin_rho(Y, eps=eps)

    info = {
        "mode": mode,
        "K": int(K),
        "d": int(d),
        "rho_plugin": float(rho_plugin),
        "trR2": float(trR2),
        "gamma": float(gamma),
    }

    if mode == "off":
        rho_use = 0.0
        info.update({
            "undersampling_factor": 0.0,
            "gamma_attenuation": None,
            "rho_raw": 0.0,
            "rho_use": rho_use,
            "reason": "shrinkage disabled",
        })
        return rho_use, info

    if mode == "fixed":
        rho_use = float(np.clip(rho_value, 0.0, 1.0))
        info.update({
            "undersampling_factor": max(0.0, 1.0 - K / max(d, 1)),
            "gamma_attenuation": 1.0,
            "rho_raw": float(rho_value),
            "rho_use": rho_use,
            "reason": "user-specified fixed rho",
        })
        return rho_use, info

    # ----- auto mode -----
    if K >= d:
        rho_use = float(np.clip(kgeq_d_value, 0.0, 1.0))
        info.update({
            "undersampling_factor": 0.0,
            "gamma_attenuation": None,
            "rho_raw": float(kgeq_d_value),
            "rho_use": rho_use,
            "reason": "K >= d, only numerical protection if requested",
        })
        return rho_use, info

    undersampling_factor = max(0.0, 1.0 - K / max(d, 1))
    gamma_attenuation = 1.0 / (1.0 + gamma_shrink_factor * gamma)

    rho_raw = rho_plugin * undersampling_factor * gamma_attenuation * auto_gain
    rho_use = float(np.clip(rho_raw, rho_min, rho_max))

    info.update({
        "undersampling_factor": float(undersampling_factor),
        "gamma_attenuation": float(gamma_attenuation),
        "rho_raw": float(rho_raw),
        "rho_use": rho_use,
        "reason": "auto heuristic: plugin x undersampling x gamma attenuation",
    })
    return rho_use, info


# =========================
# Method A: MM + CVX ( precision Theta )
# =========================
def regularized_tyler_mm_cvx_precision(
    Y,
    gamma=1e-2,
    max_iter=100,
    tol=1e-8,
    denom_eps=1e-12,
    verbose=True,
    solver="SCS",
):
    Y = np.asarray(Y)
    K, d = Y.shape
    I = np.eye(d)
    epsv = 1e-12
    alpha = 1.0 + gamma / d
    delta = 1e-9

    Sigma_k = np.eye(d)
    Sigma_k = normalize_trace(Sigma_k, d, eps=epsv)
    Theta_k = inv_spd(Sigma_k, eps_eig=epsv)

    objSigma_hist = []
    objTheta_hist = []
    rel_hist = []
    sur_hist = []

    for it in range(1, max_iter + 1):
        tr_theta = float(np.trace(Theta_k)) + epsv
        V = Y @ Theta_k                       # shape: (K, d)
        q = np.sum(Y * V, axis=1)             # shape: (K,)
        q = np.clip(q, 1e-18, None)           # numerical safeguard
        w = 1.0 / (q + denom_eps)             # shape: (K,)

        A_k = (d / K) * (Y.T @ (w[:, None] * Y)) + (gamma / tr_theta) * I
        A_k = 0.5 * (A_k + A_k.T)

        Theta = cp.Variable((d, d), symmetric=True)
        obj = cp.Minimize(-alpha * cp.log_det(Theta) + cp.trace(A_k @ Theta))
        constraints = [Theta >> delta * np.eye(d)]
        prob = cp.Problem(obj, constraints)

        try:
            if solver.upper() == "MOSEK":
                prob.solve(solver=cp.MOSEK, verbose=False)
            else:
                prob.solve(solver=cp.SCS, verbose=False)
            Theta_val = Theta.value
        except Exception:
            Theta_val = None

        if Theta_val is None:
            Theta_val = alpha * inv_spd(A_k, eps_eig=epsv)

        Theta_new = 0.5 * (Theta_val + Theta_val.T)

        Sigma_new = inv_spd(Theta_new, eps_eig=epsv)
        Sigma_new = normalize_trace(Sigma_new, d, eps=epsv)
        Theta_new = inv_spd(Sigma_new, eps_eig=epsv)

        rel_delta = np.linalg.norm(Sigma_new - Sigma_k, ord="fro") / (np.linalg.norm(Sigma_k, ord="fro") + epsv)
        rel_hist.append(float(rel_delta))
        sur_hist.append(float(prob.value) if prob.value is not None else np.nan)

        Sigma_k = Sigma_new
        Theta_k = Theta_new

        objSigma = regularized_objective_scatter(Sigma_k, Y, gamma=gamma, epsv=epsv)
        objTheta = regularized_objective_precision(Theta_k, Y, gamma=gamma, epsv=epsv)
        objSigma_hist.append(objSigma)
        objTheta_hist.append(objTheta)

        status = prob.status if prob.status is not None else "closed_form_fallback"
        if verbose and ((it % 10 == 0) or (it == 1)):
            print(
                f"[MM+CVX] iter {it-1:03d}  "
                f"J(Sigma)={objSigma:.6e}  rel_delta={rel_delta:.3e}  status={status}"
            )

        if rel_delta < tol:
            break

    history = {
        "obj": np.array(objSigma_hist),
        "obj_prec": np.array(objTheta_hist),
        "rel_delta": np.array(rel_hist),
        "surrogate": np.array(sur_hist),
    }
    return Sigma_k, Theta_k, history


# =========================
# Method B: Shrinkage-Stabilized CCCP
# =========================
def shrinkage_cccp_regularized_tyler_on_scatter(
    Y,
    gamma=1e-2,
    rho_mode="auto",
    rho_value=1e-3,
    max_iter=100,
    tol=1e-8,
    verbose=True,
    eps=1e-12,
    delta=1e-12,
    rho_min=0.0,
    rho_max=0.02,
    kgeq_d_value=0.0,
    auto_gain=1.0,
    gamma_shrink_factor=10.0,
):
    Y = np.asarray(Y)
    K, d = Y.shape
    I = np.eye(d)
    epsv = 1e-12

    rho_use, rho_info = select_rho(
        Y,
        gamma=gamma,
        mode=rho_mode,
        rho_value=rho_value,
        rho_min=rho_min,
        rho_max=rho_max,
        kgeq_d_value=kgeq_d_value,
        auto_gain=auto_gain,
        gamma_shrink_factor=gamma_shrink_factor,
        eps=eps,
    )

    if verbose:
        print(
            "[rho selection] "
            f"mode={rho_info['mode']}, "
            f"K={rho_info['K']}, d={rho_info['d']}, "
            f"rho_plugin={_fmt_optional_float(rho_info['rho_plugin'])}, "
            f"undersampling={_fmt_optional_float(rho_info['undersampling_factor'])}, "
            f"gamma_attn={_fmt_optional_float(rho_info['gamma_attenuation'])}, "
            f"rho_raw={_fmt_optional_float(rho_info['rho_raw'])}, "
            f"rho_use={_fmt_optional_float(rho_info['rho_use'], '.6e')}, "
            f"reason={rho_info['reason']}"
        )

    Sigma = np.eye(d)
    Sigma = normalize_trace(Sigma, d, eps=epsv)

    objSigma_hist = []
    objTheta_hist = []
    rel_hist = []

    alpha = 1.0 + gamma / d

    for it in range(1, max_iter + 1):
        invSigma = inv_spd(Sigma, eps_eig=epsv)
        tr_inv = float(np.trace(invSigma)) + epsv
        V = Y @ invSigma                    # shape: (K, d)
        q = np.sum(Y * V, axis=1)           # shape: (K,)
        q = np.clip(q, 1e-18, None)         # numerical safeguard
        w = 1.0 / (q + eps)                 # shape: (K,)

        S = (d / K) * (Y.T @ (w[:, None] * Y))
        S = 0.5 * (S + S.T)

        F_gamma = (S + (gamma / tr_inv) * I) / alpha
        F_gamma = 0.5 * (F_gamma + F_gamma.T)

        Sigma_tilde = (1.0 - rho_use) * F_gamma + rho_use * I

        Sigma_new = normalize_trace(Sigma_tilde, d, eps=epsv)
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + float(delta) * I
        Sigma_new = normalize_trace(Sigma_new, d, eps=epsv)

        rel_delta = np.linalg.norm(Sigma_new - Sigma, ord="fro") / (np.linalg.norm(Sigma, ord="fro") + epsv)
        rel_hist.append(float(rel_delta))

        Theta_new = inv_spd(Sigma_new, eps_eig=epsv)
        Theta_new = 0.5 * (Theta_new + Theta_new.T)

        objSigma = regularized_objective_scatter(Sigma_new, Y, gamma=gamma, epsv=epsv)
        objTheta = regularized_objective_precision(Theta_new, Y, gamma=gamma, epsv=epsv)

        objSigma_hist.append(objSigma)
        objTheta_hist.append(objTheta)

        if verbose and ((it % 10 == 0) or (it == 1)):
            print(
                f"[ShrinkFP] iter {it-1:03d}  "
                f"J(Sigma)={objSigma:.6e}  rel_delta={rel_delta:.3e}"
            )

        Sigma = Sigma_new
        if rel_delta < tol:
            break

    Theta = inv_spd(Sigma, eps_eig=epsv)
    Theta = 0.5 * (Theta + Theta.T)

    history = {
        "obj": np.array(objSigma_hist),
        "obj_prec": np.array(objTheta_hist),
        "rel_delta": np.array(rel_hist),
        "rho_use": float(rho_use),
        "rho_mode": rho_info["mode"],
        "rho_hat": rho_info["rho_plugin"],
        "trR2": rho_info["trR2"],
        "rho_info": rho_info,
    }

    return Sigma, Theta, history


# =========================
# Main
# =========================
def main():
    plt.close("all")
    rng = np.random.default_rng(1)

    n = 20
    K = 10 * n
    max_iter = 100
    tol = 1e-8
    verbose = True

    gamma = 1e-2

    rho_mode = "auto"  # "off", "fixed", "auto"
    rho_value = 1e-3  # only used if rho_mode == "fixed"

    # ----- Graph Laplacian -----
    p = 0.25
    _, L = make_connected_laplacian(n, p, rng)

    # basis of 1^\perp
    U = sla.null_space(np.ones((1, n)))
    d = U.shape[1]

    # ----- Sample -----
    Z = rng.standard_normal((n, K))
    R = L @ Z
    R_bar = R / (np.sqrt(np.sum(R**2, axis=0, keepdims=True)) + 1e-12)
    Y = (U.T @ R_bar).T   # K x d

    print("\n[Notation clarification]")
    print("  Theta denotes the precision matrix Theta = Sigma^{-1}.")
    print(f"  gamma = {gamma:.3e}")
    if rho_mode == "fixed":
        rho_desc = f"fixed ({rho_value:.3e})"
    else:
        rho_desc = rho_mode

    print(f"  rho mode = {rho_desc}")
    print("  This script compares:")
    print("    Method A: MM+CVX")
    print("    Method B: shrinkage-stabilized CCCP\n")

    # ----- Method A (MM+CVX) -----
    print(f"Running Method A: MM+CVX (precision) on subspace (d={d})...")
    Sigma_cvx, Theta_cvx, histA = regularized_tyler_mm_cvx_precision(
        Y,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        solver="SCS",
    )

    # ----- Method B (shrinkage cccp) -----
    print(f"\nRunning Method B: Shrinkage CCCP on scatter Sigma (d={d})...")
    Sigma_sh, Theta_sh, histB = shrinkage_cccp_regularized_tyler_on_scatter(
        Y,
        gamma=gamma,
        rho_mode=rho_mode,
        rho_value=rho_value,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        kgeq_d_value=0.0,
        rho_min=0.0,
        rho_max=0.02,
        auto_gain=1.0,
        gamma_shrink_factor=10.0,
        delta=1e-12,
    )

    Theta_cvx = 0.5 * (Theta_cvx + Theta_cvx.T)
    Theta_sh = 0.5 * (Theta_sh + Theta_sh.T)

    # ----- Targets on subspace -----
    L_sub = U.T @ L @ U
    L_sub = 0.5 * (L_sub + L_sub.T)

    Target2 = np.linalg.inv(L_sub @ L_sub)
    Target2 = 0.5 * (Target2 + Target2.T)
    Target2 = normalize_trace(Target2, d)

    Target1 = np.linalg.inv(L_sub)
    Target1 = 0.5 * (Target1 + Target1.T)
    Target1 = normalize_trace(Target1, d)

    def fro_normed(A):
        return A / (np.linalg.norm(A, ord="fro") + 1e-12)

    corr_cvx_sh = float(np.dot(fro_normed(Theta_cvx).reshape(-1), fro_normed(Theta_sh).reshape(-1)))
    relF = np.linalg.norm(Theta_cvx - Theta_sh, ord="fro") / (np.linalg.norm(Theta_cvx, ord="fro") + 1e-12)

    print("\n==================== Validation (subspace) ====================")
    print(
        f"rho_plugin={histB['rho_info']['rho_plugin']:.6f}, "
        f"rho_use={histB['rho_use']:.3e}, "
        f"mode={histB['rho_info']['mode']}, "
        f"reason={histB['rho_info']['reason']}"
    )
    print(f"corr(Theta_cvx, Theta_shrink)      : {corr_cvx_sh:.8f}")
    print(f"rel Fro gap ||A-B||/||A||          : {relF:.3e}")
    print(f"corr(Theta_cvx, (L_sub^2)^-1)      : {float(np.dot(fro_normed(Theta_cvx).ravel(), fro_normed(Target2).ravel())):.8f}")
    print(f"corr(Theta_shrink, (L_sub^2)^-1)   : {float(np.dot(fro_normed(Theta_sh).ravel(), fro_normed(Target2).ravel())):.8f}")


    M_cvx = spd_sqrt(Theta_cvx, eps_eig=1e-12)
    M_sh = spd_sqrt(Theta_sh, eps_eig=1e-12)
    M_cvx = normalize_trace(M_cvx, d)
    M_sh = normalize_trace(M_sh, d)

    print("\n==================== Sqrt-Bridge (subspace) ====================")
    print(f"corr(M_cvx, (L_sub)^-1)     : {float(np.dot(fro_normed(M_cvx).ravel(), fro_normed(Target1).ravel())):.8f}")
    print(f"corr(M_sh,  (L_sub)^-1)     : {float(np.dot(fro_normed(M_sh).ravel(), fro_normed(Target1).ravel())):.8f}")


    # =========================
    # FIGURE : Eigenvalue alignment
    # =========================
    lamL, UL = np.linalg.eigh(L_sub)
    idxLam = np.argsort(lamL)
    lamL = lamL[idxLam]
    UL = UL[:, idxLam]

    law_invlam2 = 1.0 / (lamL ** 2)
    law_invlam2 = law_invlam2 * (d / np.sum(law_invlam2))

    law_invlam = 1.0 / lamL
    law_invlam = law_invlam * (d / np.sum(law_invlam))

    th_c = np.sort(np.linalg.eigvalsh(Theta_cvx))[::-1]
    th_c = th_c * (d / np.sum(th_c))
    th_s = np.sort(np.linalg.eigvalsh(Theta_sh))[::-1]
    th_s = th_s * (d / np.sum(th_s))

    m_c = np.sort(np.linalg.eigvalsh(M_cvx))[::-1]
    m_c = m_c * (d / np.sum(m_c))
    m_s = np.sort(np.linalg.eigvalsh(M_sh))[::-1]
    m_s = m_s * (d / np.sum(m_s))

    fig2 = plt.figure()
    ax = plt.gca()
    ax.grid(True)
    ax.plot(np.arange(1, d + 1), th_c, "-o", linewidth=1.5, label=r"eig($\Sigma_{\rm cvx}^{-1}$)")
    ax.plot(np.arange(1, d + 1), th_s, "--s", linewidth=1.5, label=r"eig($\Sigma_{\rm shrink}^{-1}$)")
    ax.plot(np.arange(1, d + 1), law_invlam2, "k:", linewidth=2.0, label=r"$1/\lambda(L_{sub})^2$")
    ax.plot(np.arange(1, d + 1), m_c, "-.^", linewidth=1.5, label=r"eig($\tilde{L}^{\dagger}_{\rm cvx}$)")
    ax.plot(np.arange(1, d + 1), m_s, ":d", linewidth=1.5, label=r"eig($\tilde{L}^{\dagger}_{\rm shrink}$)")
    ax.plot(np.arange(1, d + 1), law_invlam, "k-.", linewidth=2.0, label=r"$1/\lambda(L_{sub})$")
    ax.set_yscale("log")
    ax.set_xlabel("Index (sorted)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=11, fontweight='bold')
    #ax.set_title(r"Eigenvalue on Range Space of $L$: $\Sigma^{-1}$ and $M=\sqrt{\Sigma^{-1}}$", fontsize=12, fontweight='bold')
    ax.set_title(r"Eigenvalue on Range Space of $L$ and $\tilde{L}^{\dagger}$", fontsize=12, fontweight='bold')
    ax.legend(loc="best")
    ax.tick_params(labelsize=12)
    fig2.savefig(os.path.join(save_dir, f"{script_name}_eigvals_alignment.png"), dpi=300, bbox_inches="tight")
    fig2.savefig(os.path.join(save_dir, f"{script_name}_eigvals_alignment.pdf"), bbox_inches="tight")


    # =========================
    # FIGURE : Operator action scatter
    # =========================
    Nsamp = 200
    Zt = rng.standard_normal((n, Nsamp))
    Rt = L @ Zt
    Rt = Rt / (np.sqrt(np.sum(Rt**2, axis=0, keepdims=True)) + 1e-12)
    Ytrue_full = sla.pinv(L) @ Rt

    Theta_full_cvx = U @ Theta_cvx @ U.T
    Theta_full_sh = U @ Theta_sh @ U.T
    M_full_cvx = U @ M_cvx @ U.T
    M_full_sh = U @ M_sh @ U.T

    Y_th_cvx = Theta_full_cvx @ Rt
    Y_th_sh = Theta_full_sh @ Rt
    Y_m_cvx = M_full_cvx @ Rt
    Y_m_sh = M_full_sh @ Rt

    Ytrue_sub = U.T @ Ytrue_full
    Atrue = UL.T @ Ytrue_sub
    Ath_c = UL.T @ (U.T @ Y_th_cvx)
    Ath_s = UL.T @ (U.T @ Y_th_sh)
    Am_c = UL.T @ (U.T @ Y_m_cvx)
    Am_s = UL.T @ (U.T @ Y_m_sh)

    at = Atrue.ravel()
    a_th_c = Ath_c.ravel()
    a_th_s = Ath_s.ravel()
    a_m_c = Am_c.ravel()
    a_m_s = Am_s.ravel()

    corr_th_c = pearson_corr(at, a_th_c)
    corr_th_s = pearson_corr(at, a_th_s)
    corr_m_c = pearson_corr(at, a_m_c)
    corr_m_s = pearson_corr(at, a_m_s)

    print("\n==================== Action in spectral coords ====================")
    print(f"corr(true, Theta_cvx*r)   : {corr_th_c:.6f}")
    print(f"corr(true, Theta_sh*r)    : {corr_th_s:.6f}")
    print(f"corr(true, M_cvx*r)       : {corr_m_c:.6f}")
    print(f"corr(true, M_sh*r)        : {corr_m_s:.6f}")

    fig3 = plt.figure()
    ax = plt.gca()
    ax.grid(True)
    ax.scatter(at, a_th_c, s=8, alpha=0.12, label=r"$\Sigma_{\rm cvx}^{-1} r$")
    ax.scatter(at, a_th_s, s=8, alpha=0.12, label=r"$\Sigma_{\rm shrink}^{-1} r$")
    ax.scatter(at, a_m_c, s=8, alpha=0.45, label=r"$\tilde{L}^{\dagger}_{\rm cvx} r$")
    ax.scatter(at, a_m_s, s=8, alpha=0.45, label=r"$\tilde{L}^{\dagger}_{\rm shrink} r$")
    xmin, xmax = float(np.min(at)), float(np.max(at))
    ax.plot([xmin, xmax], [xmin, xmax], "k--", linewidth=1.2, label="y=x")
    ax.set_xlabel("Spectral coords of true action $L^{\dagger}*r$", fontsize=11, fontweight='bold')
    ax.set_ylabel("Spectral coords of learned operator action", fontsize=11, fontweight='bold')
    ax.set_title(
        rf"Operator Action ($\tilde{{L}}^{{\dagger}}_{{\rm cvx}}={corr_m_c:.3f}, "
        rf"\tilde{{L}}^{{\dagger}}_{{\rm sh}}={corr_m_s:.3f}$)",
        fontsize=12,
        fontweight='bold'
    )
    #ax.set_title(f"Operator Action ($\tilde{L}^{\dagger}$_cvx={corr_m_c:.3f}, $\tilde{L}^{\dagger}$_sh={corr_m_s:.3f})", fontsize=12, fontweight='bold')
    ax.legend(loc="best")
    ax.tick_params(labelsize=12)
    fig3.savefig(os.path.join(save_dir, f"{script_name}_action_scatter.png"), dpi=300, bbox_inches="tight")
    fig3.savefig(os.path.join(save_dir, f"{script_name}_action_scatter.pdf"), bbox_inches="tight")

    print(f"\nSaved figures to folder: {save_dir}")


if __name__ == "__main__":
    main()