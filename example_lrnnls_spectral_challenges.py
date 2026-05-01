import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import pinv, eigvalsh
from pathlib import Path


plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 14,          # base font
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 2.2,
})

def style_axes(ax):
    ax.tick_params(axis="both", which="major", length=6, width=1.2, direction="out")
    ax.tick_params(axis="both", which="minor", length=3, width=1.0, direction="out")


# ---------------------------------------------------------
# 1. Problem Construction (LR-NNLS)
# ---------------------------------------------------------
n = 60
main_diag = 2 * np.ones(n)
off_diag = -1 * np.ones(n - 1)
L = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
L[0, 0] = 1
L[-1, -1] = 1

# Underdetermined sampling matrix A (m < n)
m = 25
np.random.seed(123)
indices = np.sort(np.random.choice(n, m, replace=False))
A = np.zeros((m, n))
A[np.arange(m), indices] = 1.0

M = A.T @ A + L
M_pinv = pinv(M)


# ---------------------------------------------------------
# 1.5 DATA: Generate nonnegative ground-truth + noisy samples
# ---------------------------------------------------------
rng = np.random.default_rng(7)
x_true = rng.random(n)
noise_std = 0.03
b = A @ x_true + noise_std * rng.standard_normal(m)


# ---------------------------------------------------------
# 2. File saving
# ---------------------------------------------------------
try:
    prefix = Path(__file__).stem
except NameError:
    prefix = "LR_NNLS_Hardness"


OUT_DIR = Path(f"{prefix}_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, plot_name):
    fig.tight_layout(pad=0.2)
    png_path = OUT_DIR / f"{prefix}_{plot_name}.png"
    pdf_path = OUT_DIR / f"{prefix}_{plot_name}.pdf"
    fig.savefig(png_path, format="png")
    fig.savefig(pdf_path, format="pdf")
    print(f"Saved: {png_path} and {pdf_path}")


# =========================================================
# 3. Visualizations
# =========================================================
FIGSIZE = (6.2, 5.2)

# --- Figure 1: Density of pseudoinverse ---
fig1, ax1 = plt.subplots(figsize=FIGSIZE)
ax1.spy(M_pinv, precision=1e-3, markersize=2.5, color="crimson", marker="s")
#ax1.set_title(r"Density of $(A^\top A + L)^{\dagger}$", fontsize=12, fontweight='bold')
ax1.set_xlabel("Columns", fontsize=11, fontweight='bold')
ax1.set_ylabel("Rows", fontsize=11, fontweight='bold')
style_axes(ax1)
save_plot(fig1, "density_inv")


evs = np.sort(eigvalsh(M))


# --- Figure 2: True convergence on LR-NNLS via PGD ---
def obj_val(x):
    r = A @ x - b
    return 0.5 * (r @ r) + 0.5 * (x @ (L @ x))

def grad_val(x):
    return A.T @ (A @ x - b) + L @ x

def proj_nonneg(x):
    return np.maximum(x, 0.0)

def grad_mapping_norm(x, step):
    g = grad_val(x)
    x_next = proj_nonneg(x - step * g)
    return np.linalg.norm((x - x_next) / step)

Lf = float(np.max(evs))
step = 0.95 / Lf

max_iters = 3000
x = np.zeros(n)
obj_hist = np.zeros(max_iters)
gmap_hist = np.zeros(max_iters)

for k in range(max_iters):
    obj_hist[k] = obj_val(x)
    gmap_hist[k] = grad_mapping_norm(x, step)
    x = proj_nonneg(x - step * grad_val(x))


obj_gap = obj_hist - np.min(obj_hist) + 1e-16
gmap = gmap_hist + 1e-16

fig2, ax2 = plt.subplots(figsize=FIGSIZE)
ax2.semilogy(obj_gap, label="Objective gap")
ax2.semilogy(gmap, label=r"Projected gradient mapping $\|G_\eta(x)\|$")
#ax2.set_title(r"LR-NNLS Convergence on Graph Signal Inpainting (PGD)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Iteration", fontsize=11, fontweight='bold')
ax2.set_ylabel("Log-scale value", fontsize=11, fontweight='bold')
ax2.grid(True, which="major", linestyle="--", alpha=0.45)
style_axes(ax2)

ax2.legend(loc="upper right", frameon=True, borderpad=0.5, handlelength=2.0)
save_plot(fig2, "convergence")

plt.show()
