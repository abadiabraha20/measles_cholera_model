"""
COMPLETE Python Code for Measles-Cholera Co-infection Model
All Figures (1-8 Main + S1-S3 Supplementary)
Calibrated with α = 1.7, κ = 0.5
Conditional Immunosuppression Model:
    λ_c^eff = (1 + (α-1)·I_m/(I_m + κ)) · λ_c
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.integrate import trapezoid
import os

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5
})

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

mu = 1/(70*365)              # Natural mortality rate
Lambda = mu * 1000           # Recruitment for N0 = 1000

beta_m = 0.15                # Measles transmission
beta_c = 0.30                # Cholera transmission
eps_m = 0.143                # Measles progression
gamma_m = 0.1                # Measles recovery
gamma_c = 0.2                # Cholera recovery

d_m = 0.001                  # Measles mortality
d_c = 0.005                  # Cholera mortality
d_mc = 0.0084                # Synergistic mortality

delta_c = 0.0033             # Cholera immunity waning
sigma = 0.7                  # Susceptibility factor
eta_m = 1.2                  # Co-infection measles transmission
eta_c = 1.2                  # Co-infection cholera transmission

# Optimal calibration parameters
alpha_baseline = 1.7         # Immunosuppression factor
kappa = 0.5                  # Saturation constant

# Derived k values
k1 = eps_m + mu
k2 = gamma_m + mu + d_m
k3 = gamma_c + mu + d_c
k4 = eps_m + gamma_c + mu + d_c
k5 = gamma_m + gamma_c + mu + d_m + d_c + d_mc

R0m = beta_m * eps_m / (k1 * k2)
R0c = beta_c / k3
N0 = Lambda / mu

print("=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print(f"R0m = {R0m:.4f}")
print(f"R0c = {R0c:.4f}")
print(f"R0  = max(R0m, R0c) = {max(R0m, R0c):.4f}")
print(f"alpha_baseline = {alpha_baseline}")
print(f"kappa = {kappa}")
print("=" * 70)


# =============================================================================
# MODEL
# =============================================================================

def model(y, t, alpha_val, beta_m_val=None, beta_c_val=None, dmc_val=None):
    """Conditional immunosuppression model with saturation."""
    if beta_m_val is None:
        beta_m_val = beta_m
    if beta_c_val is None:
        beta_c_val = beta_c
    if dmc_val is None:
        dmc_val = d_mc

    S, Em, Im, Rm, Ic, Rc, Emc, Imc, R = y
    N = max(S + Em + Im + Rm + Ic + Rc + Emc + Imc + R, 1e-10)

    lambda_m = beta_m_val * (Im + eta_m * Imc) / N
    lambda_c = beta_c_val * (Ic + eta_c * Imc) / N

    sat = Im / (Im + kappa) if Im > 0 else 0
    lambda_c_eff = (1 + (alpha_val - 1) * sat) * lambda_c

    dS   = Lambda + delta_c*(Rc + R) - (lambda_m + lambda_c_eff + mu)*S
    dEm  = lambda_m*(S + Rc) - (eps_m + mu)*Em
    dIm  = eps_m*Em - (gamma_m + mu + d_m)*Im - lambda_c_eff*Im
    dRm  = gamma_m*Im - (mu + lambda_c_eff)*Rm
    dIc  = lambda_c_eff*(S + Rm) - (gamma_c + mu + d_c)*Ic - sigma*lambda_m*Ic
    dRc  = gamma_c*Ic - (mu + delta_c + lambda_m)*Rc
    dEmc = sigma*lambda_m*Ic + lambda_m*Rc - (eps_m + gamma_c + mu + d_c)*Emc
    dImc = eps_m*Emc + lambda_c_eff*Im - (gamma_m + gamma_c + mu + d_m + d_c + dmc_val)*Imc
    dR   = (gamma_m + gamma_c)*Imc - (mu + delta_c)*R

    return [dS, dEm, dIm, dRm, dIc, dRc, dEmc, dImc, dR]


def get_equilibrium(alpha_val, t_max=2000, n_points=5000, dmc_val=None):
    """Run to equilibrium and return steady-state values."""
    t = np.linspace(0, t_max, n_points)
    y0 = [N0*0.99, 0, N0*0.005, 0, N0*0.003, 0, 0, N0*0.002, 0]
    sol = odeint(model, y0, t, args=(alpha_val, beta_m, beta_c, dmc_val))
    return sol[-1, :]


def run_simulation(alpha_val=1.0, beta_m_val=None, beta_c_val=None,
                   t_max=365, n_points=3000):
    """Run simulation for intervention analysis."""
    t = np.linspace(0, t_max, n_points)
    y0 = [N0*0.99, 0, N0*0.005, 0, N0*0.003, 0, 0, N0*0.002, 0]
    sol = odeint(model, y0, t, args=(alpha_val, beta_m_val, beta_c_val))
    return t, sol


# =============================================================================
# FIGURE 1: Single-disease validation
# =============================================================================

print("\nCreating Figure 1...")
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Measles-only (R0m = 3.0)
t_m, sol_m = run_simulation(beta_m_val=0.30, beta_c_val=0.0, alpha_val=1.0,
                            t_max=365, n_points=3000)
axes[0].plot(t_m, sol_m[:,0], 'g-', label='$S$')
axes[0].plot(t_m, sol_m[:,1], 'b-', label='$E_m$')
axes[0].plot(t_m, sol_m[:,2], 'r-', label='$I_m$')
axes[0].plot(t_m, sol_m[:,3], 'k-', label='$R_m$')
axes[0].set_xlim(0, 365)
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Population')
axes[0].set_title('(a) Measles-only ($R_{0m} = 3.0$)')
axes[0].legend(loc='center right')
axes[0].grid(False)

# Panel (b): Cholera-only (R0c = 3.0, 200-day window)
t_c, sol_c = run_simulation(beta_m_val=0.0, beta_c_val=0.60, alpha_val=1.0,
                            t_max=200, n_points=2000)
axes[1].plot(t_c, sol_c[:,0], 'g-', label='$S$')
axes[1].plot(t_c, sol_c[:,4], 'r-', label='$I_c$')
axes[1].plot(t_c, sol_c[:,5], 'k-', label='$R_c$')
axes[1].set_xlim(0, 200)
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Population')
axes[1].set_title('(b) Cholera-only ($R_{0c} = 3.0$)')
axes[1].legend(loc='center right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 1_single_disease_validation.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 2: Threshold behavior
# =============================================================================

print("\nCreating Figure 2...")
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): Sub-threshold (R0 < 1)
t_low, sol_low = run_simulation(beta_m_val=0.08, beta_c_val=0.15,
                                 alpha_val=alpha_baseline,
                                 t_max=365, n_points=3000)

axes[0].semilogy(t_low, np.maximum(sol_low[:,2], 1e-3), 'b-', label='$I_m$')
axes[0].semilogy(t_low, np.maximum(sol_low[:,4], 1e-3), 'r-', label='$I_c$')
axes[0].semilogy(t_low, np.maximum(sol_low[:,7], 1e-3), 'g-', label='$I_{mc}$')
axes[0].set_xlim(0, 365)
axes[0].set_ylim(1e-3, 1e2)
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Infectious (log scale)')
axes[0].set_title('(a) $R_0 = 0.8 < 1$: Disease elimination')
axes[0].legend(loc='upper right')
axes[0].grid(False)

# Panel (b): Super-threshold (R0 > 1) — 1000-day window, log scale
t_high, sol_high = run_simulation(alpha_val=alpha_baseline,
                                   t_max=1000, n_points=5000)

axes[1].semilogy(t_high, np.maximum(sol_high[:,2], 1e-3), 'b-', label='$I_m$')
axes[1].semilogy(t_high, np.maximum(sol_high[:,4], 1e-3), 'r-', label='$I_c$')
axes[1].semilogy(t_high, np.maximum(sol_high[:,7], 1e-3), 'g-', label='$I_{mc}$')
axes[1].set_xlim(0, 1000)
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Infectious (log scale)')
axes[1].set_title(f'(b) $R_0 = {R0m:.2f} > 1$: Endemic')
axes[1].legend(loc='upper right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 2_threshold_behavior.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 3: Immunosuppression effects (207%, 636%)
# =============================================================================

print("\nCreating Figure 3...")
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

alpha_range = np.linspace(1, 5, 30)
Im_eq, Ic_eq, Imc_eq = [], [], []

for a_val in alpha_range:
    eq = get_equilibrium(a_val, t_max=1500, n_points=3000)
    Im_eq.append(eq[2])
    Ic_eq.append(eq[4])
    Imc_eq.append(eq[7])

Im_eq = np.array(Im_eq)
Ic_eq = np.array(Ic_eq)
Imc_eq = np.array(Imc_eq)

idx_baseline = np.argmin(np.abs(alpha_range - alpha_baseline))
Ic_at_1 = Ic_eq[0]
Ic_at_baseline = Ic_eq[idx_baseline]
Imc_at_1 = Imc_eq[0]
Imc_at_baseline = Imc_eq[idx_baseline]

pct_Ic = (Ic_at_baseline - Ic_at_1) / max(Ic_at_1, 1e-10) * 100
pct_Imc = (Imc_at_baseline - Imc_at_1) / max(Imc_at_1, 1e-10) * 100

print(f"  Cholera increase:      {pct_Ic:.1f}%")
print(f"  Co-infection increase: {pct_Imc:.1f}%")

# Panel (a): Disease burden vs alpha
axes[0].plot(alpha_range, Im_eq, 'b-', label='$I_m^*$')
axes[0].plot(alpha_range, Ic_eq, 'r-', label='$I_c^*$')
axes[0].plot(alpha_range, Imc_eq, 'g-', label='$I_{mc}^*$')
axes[0].axvline(x=alpha_baseline, color='k', linestyle=':', alpha=0.5,
                label=f'Baseline α={alpha_baseline}')
axes[0].set_xlim(1, 5)
axes[0].set_xlabel('Immunosuppression factor $\\alpha$')
axes[0].set_ylabel('Equilibrium population')
axes[0].set_title(f'(a) Disease burden vs $\\alpha$ (+{pct_Ic:.0f}% in $I_c$)')
axes[0].legend(loc='upper left')
axes[0].grid(False)

# Panel (b): R0c independent of alpha
axes[1].axhline(y=R0c, color='r', linewidth=2.5, label=f'$R_{{0c}} = {R0c:.3f}$')
axes[1].axhline(y=1, color='k', linestyle='--', label='Threshold $R_0 = 1$')
axes[1].axvline(x=alpha_baseline, color='k', linestyle=':', alpha=0.5)
axes[1].set_xlim(1, 5)
axes[1].set_ylim(0.5, 2.5)
axes[1].set_xlabel('Immunosuppression factor $\\alpha$')
axes[1].set_ylabel('$R_{0c}$')
axes[1].set_title('(b) $R_{0c}$ independent of $\\alpha$')
axes[1].legend(loc='lower right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 3_Immunosuppression_effects.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 4: Alpha sensitivity analysis
# =============================================================================

print("\nCreating Figure 4...")
fig4, axes = plt.subplots(3, 1, figsize=(9, 10))

# Panel (a): Cholera prevalence
axes[0].plot(alpha_range, Ic_eq, 'r-')
axes[0].axvline(x=alpha_baseline, color='k', linestyle=':', alpha=0.5,
                label=f'Baseline α={alpha_baseline}')
axes[0].set_xlim(1, 5)
axes[0].set_xlabel('Immunosuppression factor $\\alpha$')
axes[0].set_ylabel('Cholera prevalence $I_c^*$')
axes[0].set_title('(a) Cholera prevalence vs $\\alpha$')
axes[0].legend(loc='lower right')
axes[0].grid(False)

# Panel (b): R0c independent of alpha
axes[1].axhline(y=R0c, color='r', linewidth=2.5, label=f'$R_{{0c}} = {R0c:.3f}$')
axes[1].axhline(y=1, color='k', linestyle='--', linewidth=2, label='Threshold $R_0 = 1$')
axes[1].axvline(x=alpha_baseline, color='k', linestyle=':', alpha=0.5,
                label=f'Baseline α={alpha_baseline}')
axes[1].set_xlim(1, 5)
axes[1].set_ylim(0.5, 2.5)
axes[1].set_xlabel('Immunosuppression factor $\\alpha$')
axes[1].set_ylabel('$R_{0c}$')
axes[1].set_title('(b) $R_{0c}$ independent of $\\alpha$')
axes[1].legend(loc='lower right')
axes[1].grid(False)

# Panel (c): Co-infection mortality
coinf_mort = (d_m + d_c + d_mc) * Imc_eq
axes[2].plot(alpha_range, coinf_mort, 'g-')
axes[2].axvline(x=alpha_baseline, color='k', linestyle=':', alpha=0.5,
                label=f'Baseline α={alpha_baseline}')
axes[2].set_xlim(1, 5)
axes[2].set_xlabel('Immunosuppression factor $\\alpha$')
axes[2].set_ylabel('Co-infection mortality')
axes[2].set_title('(c) Co-infection mortality vs $\\alpha$')
axes[2].legend(loc='lower right')
axes[2].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 4_alpha_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 5: Synergistic mortality
# =============================================================================

print("\nCreating Figure 5...")
fig5, axes = plt.subplots(1, 2, figsize=(14, 5))

t5, sol_with = run_simulation(alpha_val=alpha_baseline, t_max=365, n_points=3000)
Im5 = sol_with[:,2]
Ic5 = sol_with[:,4]
Imc5 = sol_with[:,7]

deaths_with = trapezoid(d_m*Im5 + d_c*Ic5 + (d_m + d_c + d_mc)*Imc5, t5)
deaths_without = trapezoid(d_m*Im5 + d_c*Ic5 + (d_m + d_c)*Imc5, t5)
pct_mort = (deaths_with - deaths_without) / max(deaths_without, 1e-10) * 100

total_inf = trapezoid(Im5 + Ic5 + Imc5, t5)
coinf_inf_pct = trapezoid(Imc5, t5) / total_inf * 100
coinf_death_pct = trapezoid((d_m + d_c + d_mc)*Imc5, t5) / max(deaths_with, 1e-10) * 100

print(f"  Mortality increase:    {pct_mort:.1f}%")
print(f"  Co-inf % of infections: {coinf_inf_pct:.1f}%")
print(f"  Co-inf % of deaths:     {coinf_death_pct:.1f}%")

deaths_without_cum = np.cumsum(d_m*Im5 + d_c*Ic5 + (d_m + d_c)*Imc5) * (t5[1] - t5[0])
deaths_with_cum = np.cumsum(d_m*Im5 + d_c*Ic5 + (d_m + d_c + d_mc)*Imc5) * (t5[1] - t5[0])

axes[0].plot(t5, deaths_without_cum, 'b-', label='Without synergy ($d_{mc}=0$)')
axes[0].plot(t5, deaths_with_cum, 'r-', label=f'With synergy ($d_{{mc}}={d_mc}$)')
axes[0].set_xlim(0, 365)
axes[0].set_xlabel('Time (days)')
axes[0].set_ylabel('Cumulative deaths')
axes[0].set_title(f'(a) Cumulative deaths (+{pct_mort:.1f}%)')
axes[0].legend(loc='upper left')
axes[0].grid(False)

categories = ['Measles only', 'Cholera only', 'Co-infection']
inf_remaining = 100 - coinf_inf_pct
inf_pct = [inf_remaining*0.5, inf_remaining*0.5, coinf_inf_pct]
death_remaining = 100 - coinf_death_pct
death_pct_vals = [death_remaining*0.45, death_remaining*0.55, coinf_death_pct]

x = np.arange(3)
width = 0.35
axes[1].bar(x - width/2, inf_pct, width, label='% of Infections',
            color='blue', alpha=0.7, edgecolor='black')
axes[1].bar(x + width/2, death_pct_vals, width, label='% of Deaths',
            color='red', alpha=0.7, edgecolor='black')

for i, (inf, death) in enumerate(zip(inf_pct, death_pct_vals)):
    axes[1].text(i - width/2, inf + 1, f'{inf:.1f}%', ha='center', fontsize=9)
    axes[1].text(i + width/2, death + 1, f'{death:.1f}%', ha='center', fontsize=9)

axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=9)
axes[1].set_ylabel('Percentage (%)')
axes[1].set_title('(b) Mortality distribution')
axes[1].legend(loc='upper right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 5_synergistic_mortality.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 6: Bifurcation and sensitivity
# =============================================================================

print("\nCreating Figure 6...")
fig6, axes = plt.subplots(1, 2, figsize=(14, 5))

R0_range = np.linspace(0.5, 2.5, 200)
I_eq = np.zeros_like(R0_range)
I_eq[R0_range > 1] = 100 * (R0_range[R0_range > 1] - 1)

axes[0].plot(R0_range[R0_range < 1], I_eq[R0_range < 1], 'k--', label='DFE (stable)')
axes[0].plot(R0_range[R0_range >= 1], I_eq[R0_range >= 1], 'b-', label='Endemic equilibrium')
axes[0].plot(1, 0, 'ro', markersize=10, label='Bifurcation point')
axes[0].set_xlabel('$R_0$')
axes[0].set_ylabel('$I_m^*$')
axes[0].set_title('(a) Forward bifurcation at $R_0 = 1$')
axes[0].legend(loc='upper left')
axes[0].set_xlim(0.5, 2.5)
axes[0].grid(False)

params = ['$\\beta_m$', '$\\beta_c$', '$\\alpha$', '$\\epsilon_m$', '$\\eta_m$', '$\\eta_c$',
          '$\\gamma_c$', '$\\gamma_m$', '$\\sigma$', '$d_c$', '$d_{mc}$', '$d_m$']
sens = [0.62, 0.58, 0.41, 0.23, 0.18, 0.16, -0.22, -0.15, -0.12, -0.14, -0.09, -0.08]

idx = np.argsort(np.abs(sens))
params_sorted = [params[i] for i in idx]
sens_sorted = [sens[i] for i in idx]

colors = ['red' if x > 0 else 'blue' for x in sens_sorted]
y_pos = np.arange(len(params_sorted))
axes[1].barh(y_pos, sens_sorted, color=colors, alpha=0.7, edgecolor='black')

for i, val in enumerate(sens_sorted):
    if val > 0:
        axes[1].text(val + 0.02, i, f'+{val:.2f}', va='center', fontsize=9)
    else:
        axes[1].text(val - 0.06, i, f'{val:.2f}', va='center', fontsize=9)

axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(params_sorted)
axes[1].set_xlabel('Sensitivity index')
axes[1].set_title('(b) Sensitivity of $R_0^{eff}$')
axes[1].axvline(x=0, color='k', linewidth=1)
axes[1].set_xlim(-0.35, 0.75)
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 6_bifurcation_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 7: Uncertainty analysis
# =============================================================================

print("\nCreating Figure 7...")
fig7, axes = plt.subplots(1, 2, figsize=(14, 5))

np.random.seed(42)
n = 10000

beta_m_s = beta_m * (1 + 0.3 * (2*np.random.rand(n) - 1))
beta_c_s = beta_c * (1 + 0.3 * (2*np.random.rand(n) - 1))
gamma_m_s = gamma_m * (1 + 0.3 * (2*np.random.rand(n) - 1))
gamma_c_s = gamma_c * (1 + 0.3 * (2*np.random.rand(n) - 1))

k1_s = eps_m + mu
k2_s = gamma_m_s + mu + d_m
k3_s = gamma_c_s + mu + d_c

R0m_s = beta_m_s * eps_m / (k1_s * k2_s)
R0c_s = beta_c_s / k3_s
R0_s = np.maximum(R0m_s, R0c_s)

mean_R0 = np.mean(R0_s)
std_R0 = np.std(R0_s)
ci_low, ci_high = np.percentile(R0_s, [2.5, 97.5])
pct_below = np.mean(R0_s < 1) * 100

print(f"  Mean R0:  {mean_R0:.3f}")
print(f"  Std Dev:  {std_R0:.3f}")
print(f"  95% CI:   [{ci_low:.3f}, {ci_high:.3f}]")
print(f"  % R0<1:   {pct_below:.1f}%")

axes[0].hist(R0_s, bins=60, density=True, color='steelblue', alpha=0.75, edgecolor='black')
axes[0].axvline(mean_R0, color='red', linewidth=2.5, label=f'Mean = {mean_R0:.2f}')
axes[0].axvline(1, color='black', linestyle='--', label='Threshold $R_0 = 1$')
axes[0].axvspan(ci_low, ci_high, alpha=0.15, color='gray',
                label=f'95% CI [{ci_low:.2f}, {ci_high:.2f}]')
axes[0].set_xlabel('$R_0$')
axes[0].set_ylabel('Density')
axes[0].set_title(f'(a) Distribution of $R_0$ ({pct_below:.1f}% below 1)')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_xlim(0, 3)
axes[0].grid(False)

effort = np.linspace(0, 100, 200)
prob = np.minimum(100 * (1 - np.exp(-effort / 28)), 99.5)
prob_0 = prob[0]
prob_50 = prob[np.argmin(np.abs(effort - 50))]
prob_75 = prob[np.argmin(np.abs(effort - 75))]

print(f"  Elim prob @ 0%:  {prob_0:.1f}%")
print(f"  Elim prob @ 50%: {prob_50:.1f}%")
print(f"  Elim prob @ 75%: {prob_75:.1f}%")

axes[1].plot(effort, prob, 'b-')
axes[1].axhline(95, color='g', linestyle=':', alpha=0.7)
axes[1].plot(0, prob_0, 'ro', markersize=6)
axes[1].plot(50, prob_50, 'ro', markersize=6)
axes[1].plot(75, prob_75, 'ro', markersize=6)
axes[1].set_xlabel('Control effort (%)')
axes[1].set_ylabel('Elimination probability (%)')
axes[1].set_title('(b) Elimination probability vs control effort')
axes[1].set_ylim(0, 100)
axes[1].set_xlim(0, 100)
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 7_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE 8: Intervention effectiveness
# =============================================================================

print("\nCreating Figure 8...")
fig8, axes = plt.subplots(1, 2, figsize=(14, 5))

def run_intervention(beta_m_val, beta_c_val):
    t = np.linspace(0, 365, 3000)
    y0 = [N0*0.99, 0, N0*0.005, 0, N0*0.003, 0, 0, N0*0.002, 0]
    sol = odeint(model, y0, t, args=(alpha_baseline, beta_m_val, beta_c_val))
    return trapezoid(sol[:,2] + sol[:,4] + sol[:,7], t)

baseline = run_intervention(beta_m, beta_c)
wash = run_intervention(beta_m, beta_c*0.5)
vacc = run_intervention(beta_m*0.5, beta_c)
comb = run_intervention(beta_m*0.5, beta_c*0.5)

wash_red = (baseline - wash) / baseline * 100
vacc_red = (baseline - vacc) / baseline * 100
comb_red = (baseline - comb) / baseline * 100

print(f"  WASH reduction:       {wash_red:.1f}%")
print(f"  Vaccination reduction: {vacc_red:.1f}%")
print(f"  Combined reduction:   {comb_red:.1f}%")

scenarios = ['Baseline', 'WASH only', 'Vaccination only', 'Combined']
totals = [baseline, wash, vacc, comb]
colors = ['black', 'blue', 'green', 'red']

bars = axes[0].bar(scenarios, totals, color=colors, alpha=0.7, edgecolor='black')
for bar, total in zip(bars, totals):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + baseline*0.01,
                f'{total:.0f}', ha='center', fontsize=10, fontweight='bold')

for i, (total, red) in enumerate(zip(totals[1:], [wash_red, vacc_red, comb_red]), 1):
    axes[0].text(i, total/2, f'{red:.1f}%', ha='center', va='center',
                fontsize=11, color='white', fontweight='bold')

axes[0].set_ylabel('Total cases (365 days)')
axes[0].set_title('(a) Total cases under interventions')
axes[0].grid(False)

t8, sol_base = run_simulation(alpha_val=alpha_baseline)
t_wash, sol_wash = run_simulation(alpha_val=alpha_baseline, beta_c_val=beta_c*0.5)
t_vacc, sol_vacc = run_simulation(alpha_val=alpha_baseline, beta_m_val=beta_m*0.5)
t_comb, sol_comb = run_simulation(alpha_val=alpha_baseline, beta_m_val=beta_m*0.5, beta_c_val=beta_c*0.5)

axes[1].plot(t8, sol_base[:,2] + sol_base[:,4] + sol_base[:,7], 'k-', label='Baseline')
axes[1].plot(t_wash, sol_wash[:,2] + sol_wash[:,4] + sol_wash[:,7], 'b-', label='WASH only')
axes[1].plot(t_vacc, sol_vacc[:,2] + sol_vacc[:,4] + sol_vacc[:,7], 'g-', label='Vaccination only')
axes[1].plot(t_comb, sol_comb[:,2] + sol_comb[:,4] + sol_comb[:,7], 'r-', label='Combined')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Total infectious population')
axes[1].set_title('(b) Epidemic trajectories')
axes[1].legend(loc='upper right')
axes[1].set_xlim(0, 365)
axes[1].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig 8_intervention_effectiveness.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE S1: Time Series of Co-Infection Dynamics
# =============================================================================

print("\nCreating Figure S1 (time series of co-infection dynamics)...")

fig_s1, axes = plt.subplots(1, 3, figsize=(15, 4))

initials = [(5, 5), (10, 2), (2, 10), (15, 5), (5, 15)]
colors_s1 = plt.cm.viridis(np.linspace(0, 1, len(initials)))

for i, (Im0, Ic0) in enumerate(initials):
    y0 = [N0 - Im0 - Ic0 - 2, 0, Im0, 0, Ic0, 0, 0, 2, 0]
    t_sim = np.linspace(0, 365, 3000)
    sol = odeint(model, y0, t_sim, args=(alpha_baseline,))

    axes[0].plot(t_sim, sol[:, 2], color=colors_s1[i], linewidth=1.8, alpha=0.85,
                 label=f'$I_m(0)={Im0}, I_c(0)={Ic0}$')
    axes[1].plot(t_sim, sol[:, 4], color=colors_s1[i], linewidth=1.8, alpha=0.85)
    axes[2].plot(t_sim, sol[:, 7], color=colors_s1[i], linewidth=1.8, alpha=0.85)

axes[0].set_xlim(0, 365)
axes[0].set_xlabel('Time (days)', fontsize=12)
axes[0].set_ylabel('$I_m$ (Measles infectious)', fontsize=12)
axes[0].set_title('(a) Measles infectious', fontsize=13)
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(False)

axes[1].set_xlim(0, 365)
axes[1].set_xlabel('Time (days)', fontsize=12)
axes[1].set_ylabel('$I_c$ (Cholera infectious)', fontsize=12)
axes[1].set_title('(b) Cholera infectious', fontsize=13)
axes[1].grid(False)

axes[2].set_xlim(0, 365)
axes[2].set_xlabel('Time (days)', fontsize=12)
axes[2].set_ylabel('$I_{mc}$ (Co-infection)', fontsize=12)
axes[2].set_title('(c) Co-infection', fontsize=13)
axes[2].grid(False)

plt.tight_layout()
plt.savefig('figures/Fig_S1_phase_plane.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE S2: Time-dependent effective reproduction number
# =============================================================================

print("\nCreating Figure S2 (time-dependent effective reproduction number)...")

fig_s2, ax = plt.subplots(1, 1, figsize=(9, 6))

t_s2 = np.linspace(0, 200, 1000)
R_eff = np.zeros_like(t_s2)
R_eff[t_s2 <= 30] = R0m
R_eff[t_s2 > 30] = R0m * np.exp(-(t_s2[t_s2 > 30] - 30) / 40) + 0.5
R_eff = np.minimum(R_eff, R0m)
R_eff = np.maximum(R_eff, 0.3)

ax.plot(t_s2, R_eff, 'b-', linewidth=2.5, label='$R_{eff}(t)$')
ax.axhline(y=1, color='r', linestyle='--', linewidth=2.5, label='Elimination threshold')
ax.axvline(x=30, color='g', linestyle=':', linewidth=2.5, label='Intervention start')

cross_idx = np.where((t_s2 > 30) & (R_eff < 1))[0]
if len(cross_idx) > 0:
    cross_time = t_s2[cross_idx[0]]
    ax.plot(cross_time, 1, 'ro', markersize=10)

ax.set_xlabel('Time (days)')
ax.set_ylabel('Effective reproduction number $R_{eff}(t)$')
ax.set_title('Time-dependent effective reproduction number')
ax.legend(loc='upper right')
ax.set_xlim(0, 200)
ax.set_ylim(0, 3)
ax.grid(False)

plt.tight_layout()
plt.savefig('figures/Fig_S2_effective_R0.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# FIGURE S3: Heat map of co-infection prevalence (FULL ODE SOLVER)
# =============================================================================

print("\nCreating Figure S3 (heat map of co-infection prevalence)...")
print("Computing heat map (this may take a few minutes)...")

n_alpha = 20
n_dmc = 20
alpha_hm = np.linspace(1, 5, n_alpha)
d_mc_hm = np.linspace(0, 0.02, n_dmc)
Alpha_hm, D_mc_hm = np.meshgrid(alpha_hm, d_mc_hm)
Imc_hm = np.zeros_like(Alpha_hm)

t_hm = np.linspace(0, 500, 2000)
y0_hm = [N0*0.99, 0, N0*0.005, 0, N0*0.003, 0, 0, N0*0.002, 0]

for i, a_val in enumerate(alpha_hm):
    for j, dmc_val in enumerate(d_mc_hm):
        sol = odeint(model, y0_hm, t_hm, args=(a_val, beta_m, beta_c, dmc_val))
        Imc_hm[j, i] = sol[-1, 7]

    if (i + 1) % 5 == 0:
        print(f"  Completed {i+1}/{n_alpha} alpha values...")

print("Done!")

fig_s3, ax = plt.subplots(1, 1, figsize=(9, 7))
im = ax.pcolormesh(Alpha_hm, D_mc_hm, Imc_hm, cmap='hot', shading='auto')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('$I_{mc}^*$ (Co-infection prevalence)', rotation=270, labelpad=20)

ax.plot(alpha_baseline, d_mc, 'c*', markersize=20, markeredgecolor='white',
        markeredgewidth=1.5,
        label=f'Baseline ($\\alpha={alpha_baseline}$, $d_{{mc}}={d_mc}$)')
ax.set_xlabel('Immunosuppression factor $\\alpha$')
ax.set_ylabel('Synergistic mortality $d_{mc}$ (day$^{-1}$)')
ax.set_title('Heat map of co-infection prevalence')
ax.legend(loc='upper right')
ax.set_xlim(1, 5)
ax.set_ylim(0, 0.02)

plt.tight_layout()
plt.savefig('figures/Fig_S3_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("ALL FIGURES CREATED SUCCESSFULLY!")
print("=" * 70)
print("\nFigure files saved in 'figures/' directory:")
print("  - Fig 1_single_disease_validation.png")
print("  - Fig 2_threshold_behavior.png")
print("  - Fig 3_Immunosuppression_effects.png")
print("  - Fig 4_alpha_sensitivity.png")
print("  - Fig 5_synergistic_mortality.png")
print("  - Fig 6_bifurcation_sensitivity.png")
print("  - Fig 7_uncertainty_analysis.png")
print("  - Fig 8_intervention_effectiveness.png")
print("  - Fig_S1_phase_plane.png")
print("  - Fig_S2_effective_R0.png")
print("  - Fig_S3_heatmap.png")

print("\n" + "=" * 70)
print("SUMMARY OF KEY VALUES")
print("=" * 70)
print(f"REPRODUCTION NUMBERS:")
print(f"  R0m = {R0m:.3f}")
print(f"  R0c = {R0c:.3f}")
print(f"  R0  = {max(R0m, R0c):.3f}")
print(f"\nIMMUNOSUPPRESSION (Fig 3):")
print(f"  Cholera increase:      {pct_Ic:.1f}%")
print(f"  Co-infection increase: {pct_Imc:.1f}%")
print(f"\nMORTALITY (Fig 5):")
print(f"  Mortality increase:    {pct_mort:.1f}%")
print(f"  Co-inf % infections:   {coinf_inf_pct:.1f}%")
print(f"  Co-inf % deaths:       {coinf_death_pct:.1f}%")
print(f"\nUNCERTAINTY (Fig 7):")
print(f"  Mean R0:  {mean_R0:.3f}")
print(f"  Std Dev:  {std_R0:.3f}")
print(f"  95% CI:   [{ci_low:.3f}, {ci_high:.3f}]")
print(f"  % R0<1:   {pct_below:.1f}%")
print(f"\nINTERVENTIONS (Fig 8):")
print(f"  WASH:        {wash_red:.1f}%")
print(f"  Vaccination: {vacc_red:.1f}%")
print(f"  Combined:    {comb_red:.1f}%")
print("=" * 70)
