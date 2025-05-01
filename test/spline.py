import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# ── 1.  Raw data ────────────────────────────────────────────────────────────────
# knot (τ, u) pairs – chosen to match the visual shape in the reference image
tau = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
u = np.array([0.25, 0.9, -0.1, 0.0, 4.0, 0.0, 0.25])

# a dense grid for smooth curves
tau_dense = np.linspace(tau.min(), tau.max(), 1000)

# ── 2.  Interpolants ────────────────────────────────────────────────────────────
u_zero = interp1d(tau, u, kind="previous")(tau_dense)  # zero-order hold
u_linear = interp1d(tau, u, kind="linear")(tau_dense)  # piece-wise linear
u_cubic = CubicSpline(tau, u)(tau_dense)  # natural cubic spline

# ── 3.  Figure layout & style ───────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)

fig, ax = plt.subplots(figsize=(6, 5), dpi=100)

# cubic spline
ax.plot(tau_dense, u_cubic, lw=3, color="#D1E7BE", label="Cubic")
# linear
ax.plot(tau_dense, u_linear, lw=3, color="#F48B96", label="Linear")
# zero-order hold (drawn as a continuous step sequence to match the figure)
ax.step(tau_dense, u_zero, lw=4, color="#90CCEB", where="post", label="Zero")
# knots
ax.scatter(tau, u, s=80, color="#545454", zorder=5, label="Knots")

# ── 4.  Cosmetics ───────────────────────────────────────────────────────────────
ax.grid(True, linestyle="-", color="0.85")
ax.set_xlim(tau.min() - 0.3, tau.max() + 0.3)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$u$", rotation=0, labelpad=15)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# put legend in the empty space at bottom-left, in the same order as the image
handles, labels = ax.get_legend_handles_labels()
order = [2, 1, 0, 3]  # Zero, Linear, Cubic, Knots
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="lower left",
    frameon=False,
)

plt.tight_layout()
# Save figure as EPS file
plt.savefig("spline_interpolation.eps", format="eps", bbox_inches="tight")
plt.show()
