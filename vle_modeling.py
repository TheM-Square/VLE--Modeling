"""
Vapor-Liquid Equilibrium Modeling for Binary Mixtures
======================================================
Author: Manas Mahajan | NIT Raipur | Chemical Engineering
GitHub: github.com/TheM-Square

Models VLE behavior using:
- Antoine Equation for vapor pressures
- Raoult's Law (ideal systems)
- Margules Activity Coefficient Model (non-ideal systems)

Systems modeled:
- Benzene-Toluene (nearly ideal)
- Ethanol-Water (highly non-ideal)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

# ── Antoine Constants (log10(P/mmHg) = A - B/(C+T[°C])) ──────────────────────
ANTOINE = {
    "benzene":  {"A": 6.90565, "B": 1211.033, "C": 220.790},
    "toluene":  {"A": 6.95334, "B": 1343.943, "C": 219.377},
    "ethanol":  {"A": 8.11220, "B": 1592.864, "C": 226.184},
    "water":    {"A": 8.07131, "B": 1730.630, "C": 233.426},
}

# ── Margules Constants (A12, A21) for non-ideal systems ──────────────────────
MARGULES = {
    ("ethanol", "water"): (1.6022, 0.7947),
}


def vapor_pressure(component: str, T_C: float) -> float:
    """Antoine equation → vapor pressure in mmHg."""
    c = ANTOINE[component]
    return 10 ** (c["A"] - c["B"] / (c["C"] + T_C))


def activity_coefficients(x1: float, system: tuple, T_C: float = None) -> tuple:
    """Margules two-suffix model activity coefficients."""
    if system not in MARGULES:
        return 1.0, 1.0
    A12, A21 = MARGULES[system]
    x2 = 1.0 - x1
    ln_g1 = x2**2 * (A12 + 2*(A21 - A12)*x1)
    ln_g2 = x1**2 * (A21 + 2*(A12 - A21)*x2)
    return np.exp(ln_g1), np.exp(ln_g2)


def bubble_point_T(x1: float, comp1: str, comp2: str, P_mmHg: float = 760.0,
                    ideal: bool = True) -> float:
    """Iterate to find bubble-point temperature at fixed P."""
    system = (comp1, comp2)

    def residual(T):
        P1s = vapor_pressure(comp1, T)
        P2s = vapor_pressure(comp2, T)
        if ideal:
            return x1*P1s + (1-x1)*P2s - P_mmHg
        else:
            g1, g2 = activity_coefficients(x1, system, T)
            return x1*g1*P1s + (1-x1)*g2*P2s - P_mmHg

    return brentq(residual, 20, 200)


def dew_point_T(y1: float, comp1: str, comp2: str, P_mmHg: float = 760.0,
                 ideal: bool = True) -> float:
    """Iterate to find dew-point temperature at fixed P."""
    system = (comp1, comp2)

    def residual(T):
        P1s = vapor_pressure(comp1, T)
        P2s = vapor_pressure(comp2, T)
        if ideal:
            return 1.0 - y1*P_mmHg/P1s - (1-y1)*P_mmHg/P2s
        else:
            # estimate liquid composition iteratively
            x1_est = y1 * P_mmHg / P1s
            x1_est = max(1e-6, min(1-1e-6, x1_est))
            g1, g2 = activity_coefficients(x1_est, system, T)
            return 1.0 - y1*P_mmHg/(g1*P1s) - (1-y1)*P_mmHg/(g2*P2s)

    return brentq(residual, 20, 200)


def generate_Txy(comp1, comp2, P=760, ideal=True, n=60):
    x_arr = np.linspace(1e-4, 1-1e-4, n)
    T_bub, T_dew = [], []
    for x1 in x_arr:
        T_bub.append(bubble_point_T(x1, comp1, comp2, P, ideal))
        T_dew.append(dew_point_T(x1, comp1, comp2, P, ideal))
    return x_arr, np.array(T_bub), np.array(T_dew)


def generate_Pxy(comp1, comp2, T_C=80, ideal=True, n=60):
    x_arr = np.linspace(0, 1, n)
    system = (comp1, comp2)
    P1s = vapor_pressure(comp1, T_C)
    P2s = vapor_pressure(comp2, T_C)
    P_bub, P_dew = [], []
    for x1 in x_arr:
        if ideal:
            pb = x1*P1s + (1-x1)*P2s
        else:
            g1, g2 = activity_coefficients(x1, system, T_C)
            pb = x1*g1*P1s + (1-x1)*g2*P2s
        P_bub.append(pb)
        y1 = x1*P1s/pb if ideal else x1*g1*P1s/pb
        y1 = max(0, min(1, y1))
        P_dew.append(P_mmHg_from_y(y1, comp1, comp2, T_C, ideal, system))
    return x_arr, np.array(P_bub), np.array(P_dew)


def P_mmHg_from_y(y1, comp1, comp2, T_C, ideal, system):
    P1s = vapor_pressure(comp1, T_C)
    P2s = vapor_pressure(comp2, T_C)
    if ideal:
        denom = y1/P1s + (1-y1)/P2s
        return 1.0/denom if denom > 0 else 760
    else:
        x1_est = max(1e-6, min(1-1e-6, y1))
        g1, g2 = activity_coefficients(x1_est, system, T_C)
        denom = y1/(g1*P1s) + (1-y1)/(g2*P2s)
        return 1.0/denom if denom > 0 else 760


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all():
    plt.style.use("seaborn-v0_8-whitegrid")
    COLORS = {
        "bubble_ideal":    "#2563EB",
        "dew_ideal":       "#7C3AED",
        "bubble_nonideal": "#DC2626",
        "dew_nonideal":    "#EA580C",
        "fill_ideal":      "#DBEAFE",
        "fill_nonideal":   "#FEE2E2",
    }

    fig = plt.figure(figsize=(18, 14), facecolor="#0F172A")
    fig.suptitle(
        "Vapor–Liquid Equilibrium Modeling for Binary Mixtures",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1E293B")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color="#94A3B8")
        ax.set_ylabel(ylabel, fontsize=9, color="#94A3B8")
        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(color="#334155", linewidth=0.5)

    # ── 1. Benzene-Toluene T-xy (ideal) ──
    x, Tb, Td = generate_Txy("benzene", "toluene", ideal=True)
    ax = axes[0]
    ax.fill_between(x, Tb, Td, alpha=0.25, color=COLORS["fill_ideal"])
    ax.plot(x, Tb, color=COLORS["bubble_ideal"], lw=2.2, label="Bubble point")
    ax.plot(x, Td, color=COLORS["dew_ideal"],    lw=2.2, label="Dew point", ls="--")
    ax.annotate("Two-phase\nregion", xy=(0.5, (Tb[30]+Td[30])/2),
                fontsize=8, color="#93C5FD", ha="center")
    ax.legend(fontsize=8, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "Benzene–Toluene  T–xy  (Ideal, 1 atm)",
             "x₁, y₁  (Benzene mole fraction)", "Temperature (°C)")

    # ── 2. Ethanol-Water T-xy (ideal vs non-ideal) ──
    x_i, Tb_i, Td_i = generate_Txy("ethanol", "water", ideal=True)
    x_n, Tb_n, Td_n = generate_Txy("ethanol", "water", ideal=False)
    ax = axes[1]
    ax.plot(x_i, Tb_i, color=COLORS["bubble_ideal"], lw=1.8, ls="--", label="Bubble (Raoult)")
    ax.plot(x_i, Td_i, color=COLORS["dew_ideal"],    lw=1.8, ls="--", label="Dew (Raoult)")
    ax.plot(x_n, Tb_n, color=COLORS["bubble_nonideal"], lw=2.2, label="Bubble (Margules)")
    ax.plot(x_n, Td_n, color=COLORS["dew_nonideal"],    lw=2.2, label="Dew (Margules)", ls="-.")
    ax.legend(fontsize=7.5, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "Ethanol–Water  T–xy  (Ideal vs Non-ideal, 1 atm)",
             "x₁, y₁  (Ethanol mole fraction)", "Temperature (°C)")

    # ── 3. Benzene-Toluene P-xy ──
    x, Pb, Pd = generate_Pxy("benzene", "toluene", T_C=80, ideal=True)
    ax = axes[2]
    ax.fill_between(x, Pb, Pd, alpha=0.2, color=COLORS["fill_ideal"])
    ax.plot(x, Pb, color=COLORS["bubble_ideal"], lw=2.2, label="Bubble (liquid)")
    ax.plot(x, Pd, color=COLORS["dew_ideal"],    lw=2.2, label="Dew (vapor)", ls="--")
    ax.axhline(760, color="#FBBF24", lw=1, ls=":", label="1 atm")
    ax.legend(fontsize=8, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "Benzene–Toluene  P–xy  (80°C)",
             "x₁, y₁  (Benzene mole fraction)", "Pressure (mmHg)")

    # ── 4. Activity coefficients vs composition ──
    x_arr = np.linspace(1e-4, 1-1e-4, 200)
    g1_arr = [activity_coefficients(x, ("ethanol","water"))[0] for x in x_arr]
    g2_arr = [activity_coefficients(x, ("ethanol","water"))[1] for x in x_arr]
    ax = axes[3]
    ax.plot(x_arr, g1_arr, color="#F472B6", lw=2.2, label="γ₁ (Ethanol)")
    ax.plot(x_arr, g2_arr, color="#34D399", lw=2.2, label="γ₂ (Water)")
    ax.axhline(1.0, color="#64748B", lw=1, ls="--", label="Ideal (γ=1)")
    ax.legend(fontsize=8, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "Margules Activity Coefficients  –  Ethanol–Water",
             "x₁  (Ethanol mole fraction)", "Activity Coefficient  γ")

    # ── 5. Vapor pressure vs Temperature (Antoine) ──
    T_range = np.linspace(40, 120, 200)
    ax = axes[4]
    colors_comp = {"benzene": "#60A5FA", "toluene": "#A78BFA",
                   "ethanol": "#F87171", "water": "#34D399"}
    for comp, col in colors_comp.items():
        Pvap = [vapor_pressure(comp, T) for T in T_range]
        ax.plot(T_range, Pvap, color=col, lw=2, label=comp.capitalize())
    ax.axhline(760, color="#FBBF24", lw=1.2, ls="--", label="1 atm (760 mmHg)")
    ax.legend(fontsize=8, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "Antoine Equation  –  Vapor Pressure vs Temperature",
             "Temperature (°C)", "Vapor Pressure (mmHg)")

    # ── 6. y-x McCabe-Thiele style diagram ──
    x_bt, Tb_bt, _ = generate_Txy("benzene", "toluene", ideal=True, n=80)
    y_bt = []
    for i, x1 in enumerate(x_bt):
        P1s = vapor_pressure("benzene", Tb_bt[i])
        y_bt.append(x1 * P1s / 760.0)

    x_ew, Tb_ew, _ = generate_Txy("ethanol", "water", ideal=False, n=80)
    y_ew = []
    for i, x1 in enumerate(x_ew):
        P1s = vapor_pressure("ethanol", Tb_ew[i])
        g1, _ = activity_coefficients(x1, ("ethanol","water"), Tb_ew[i])
        P_bub = x1*g1*P1s
        y_ew.append(min(P_bub / 760.0, 1.0))

    ax = axes[5]
    diag = np.linspace(0, 1, 100)
    ax.plot(diag, diag, color="#64748B", lw=1, ls="--", label="y = x (diagonal)")
    ax.plot(x_bt, y_bt, color=COLORS["bubble_ideal"], lw=2.2, label="Benzene–Toluene")
    ax.plot(x_ew, y_ew, color=COLORS["bubble_nonideal"], lw=2.2, label="Ethanol–Water")
    ax.legend(fontsize=8, facecolor="#1E293B", labelcolor="white", framealpha=0.8)
    style_ax(ax, "y–x Diagram  (McCabe–Thiele)",
             "x₁  (liquid mole fraction)", "y₁  (vapor mole fraction)")

    plt.savefig("vle_plots.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    print("Saved: vle_plots.png")
    return fig


if __name__ == "__main__":
    plot_all()
    plt.show()
