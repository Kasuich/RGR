import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

# =======================
# Config
# =======================
PATH = "/Users/kasuich/Projects/Study/University/Masters/Современные методы решения инженерных задач/РГР/РГР_расчеты_актуальные.xlsx"
SHEET = "Sheet1"

# Base time limits (hours)
T1, T2 = 3.0, 5.0

# Corridor ±10%
CORRIDOR = 0.05

# Grid for concessions
TIME_SLACKS = np.linspace(0, 70, 100)  # ΔT (hours), applied to BOTH aggregates
RES_SLACKS = np.linspace(0, 1000, 300)  # ΔR, applied to (R0 + ΔR)

# Numeric tolerance
EPS = 1e-6

# LP "no cap" value
BIG_CAP = 1e9


# =======================
# Load data
# =======================
df = pd.read_excel(PATH, sheet_name=SHEET, header=None, engine="openpyxl")

A = df.loc[3:16, 2:7].astype(float).to_numpy()  # (n, m)
P = df.loc[3:16, 9].astype(float).to_numpy()  # (n,)
agg = df.loc[3:16, 0].ffill().astype(int).to_numpy()
b = df.loc[17, 2:7].astype(float).to_numpy()  # (m,)

n, m = A.shape


# =======================
# LP helpers
# =======================
def max_lambda(time_slack, resource_cap, corridor=CORRIDOR):
    """
    Maximize lambda subject to:
      - time constraints: sum_{i in agg k} x_i / P_i <= T_k + time_slack
      - resource constraint: sum_i x_i <= resource_cap
      - corridor: (1-c)*lambda*b_j <= y_j <= (1+c)*lambda*b_j
      - x_i >= 0, 0 <= lambda <= 1
    """
    nvars = n + 1  # x_1..x_n plus lambda

    # maximize lambda <=> minimize -lambda
    c = np.zeros(nvars)
    c[-1] = -1.0

    A_ub = []
    b_ub = []

    # Time constraints
    r1 = np.zeros(nvars)
    r2 = np.zeros(nvars)
    for i in range(n):
        if agg[i] == 1:
            r1[i] = 1.0 / P[i]
        else:
            r2[i] = 1.0 / P[i]
    A_ub += [r1, r2]
    b_ub += [T1 + time_slack, T2 + time_slack]

    # Resource cap: sum x_i <= resource_cap
    r = np.zeros(nvars)
    r[:n] = 1.0
    A_ub.append(r)
    b_ub.append(resource_cap)

    # Corridor constraints
    for j in range(m):
        # lower: -y_j + (1-c)*b_j*lambda <= 0
        rr = np.zeros(nvars)
        rr[:n] = -A[:, j]
        rr[-1] = (1.0 - corridor) * b[j]
        A_ub.append(rr)
        b_ub.append(0.0)

        # upper: y_j - (1+c)*b_j*lambda <= 0
        rr = np.zeros(nvars)
        rr[:n] = A[:, j]
        rr[-1] = -(1.0 + corridor) * b[j]
        A_ub.append(rr)
        b_ub.append(0.0)

    bounds = [(0, None)] * n + [(0, 1)]

    res = linprog(
        c,
        A_ub=np.vstack(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )
    return res.x[-1] if res.success else np.nan


def min_sumx_given_lambda(lam, time_slack, corridor=CORRIDOR):
    """
    Minimize sum x subject to time constraints and corridor constraints with fixed lambda.
    """
    c = np.ones(n)

    A_ub = []
    b_ub = []

    # Time constraints
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    for i in range(n):
        if agg[i] == 1:
            r1[i] = 1.0 / P[i]
        else:
            r2[i] = 1.0 / P[i]
    A_ub += [r1, r2]
    b_ub += [T1 + time_slack, T2 + time_slack]

    # Corridor constraints with fixed lambda
    for j in range(m):
        A_ub.append(-A[:, j])
        b_ub.append(-(1.0 - corridor) * lam * b[j])

        A_ub.append(A[:, j])
        b_ub.append((1.0 + corridor) * lam * b[j])

    res = linprog(
        c,
        A_ub=np.vstack(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n,
        method="highs",
    )
    return res.fun if res.success else np.nan


# =======================
# Baseline (for ΔR axis)
# =======================
lam0 = max_lambda(time_slack=0.0, resource_cap=BIG_CAP, corridor=CORRIDOR)
if np.isnan(lam0):
    raise RuntimeError("Baseline infeasible: cannot compute lam0 with no resource cap.")

R0 = min_sumx_given_lambda(lam0, time_slack=0.0, corridor=CORRIDOR)
if np.isnan(R0):
    raise RuntimeError("Failed to compute baseline R0 (min sum x) at lam0.")

print(f"Baseline: lambda0={lam0:.6f}, baseline sum(x) R0={R0:.6f}")


# =======================
# Grid evaluation
# =======================
Z = np.zeros((len(RES_SLACKS), len(TIME_SLACKS)), dtype=float)

for yi, dR in enumerate(RES_SLACKS):
    cap = float(R0 + dR)
    for xi, dT in enumerate(TIME_SLACKS):
        Z[yi, xi] = max_lambda(
            time_slack=float(dT), resource_cap=cap, corridor=CORRIDOR
        )

grid = pd.DataFrame(
    [
        (float(dT), float(dR), float(Z[yi, xi]))
        for yi, dR in enumerate(RES_SLACKS)
        for xi, dT in enumerate(TIME_SLACKS)
    ],
    columns=["time_slack", "resource_slack", "lambda"],
)
grid.to_csv("lambda_heatmap_grid.csv", index=False)
print("Saved: lambda_heatmap_grid.csv")


# =======================
# Plot heatmap + contour levels 0.5..1.0
# =======================
fig, ax = plt.subplots(figsize=(11, 6))

im = ax.imshow(
    Z,
    origin="lower",
    aspect="auto",
    extent=[TIME_SLACKS.min(), TIME_SLACKS.max(), RES_SLACKS.min(), RES_SLACKS.max()],
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("λ (максимум)")

ax.set_xlabel("Уступка по времени ΔF_2 (часы)")
ax.set_ylabel("Уступка по ресурсам ΔF_1 (ΔΣx сверх базового F_1*)")
ax.set_title("Максимальная λ при коридоре ±5% + уровни λ (0.5..1.0)")

# Contours from 0.5 to 1.0 (step 0.1)
levels = np.arange(0.5, 1.01, 0.1)

zmax = np.nanmax(Z)
zmin = np.nanmin(Z)

if zmax < 0.5 - EPS:
    print(f"Warning: max(Z)={zmax:.3f} < 0.5; contours 0.5..1.0 will not appear.")
else:
    # keep only levels that exist in Z-range
    levels = levels[(levels >= zmin - EPS) & (levels <= zmax + EPS)]

    cmap = plt.get_cmap("Reds")
    colors = cmap(np.linspace(0.35, 0.95, len(levels)))  # light -> bright red

    CS = ax.contour(
        TIME_SLACKS,
        RES_SLACKS,
        Z,
        levels=levels,
        colors=colors,
        linewidths=0.7,  # thin
        zorder=3,  # IMPORTANT: no .collections, compatible with new matplotlib
    )

    ax.clabel(
        CS,
        inline=True,
        fontsize=9,
        fmt=lambda v: f"λ={v:.1f}",
    )

plt.tight_layout()
plt.show()
