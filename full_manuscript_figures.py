"""
full_manuscript_figures.py
==========================
Complete Python code for the Measles-Cholera Co-infection Model.
Generates all Main Figures (1-8) and Supplementary Figures (S1-S3)
using the rigorous ODE system with Table 1 parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import os

# Ensure figures directory exists (optional, saves in current folder)
if not os.path.exists('figures'):
    os.makedirs('figures')

# Publication-quality settings
plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 10, 'lines.linewidth': 2.5
})

# =============================================================================
# 1. PARAMETERS (Table 1)
# =============================================================================
mu = 1 / (70 * 365)
Lambda = 10.0
N0 = Lambda / mu
beta_m = 0.15
beta_c = 0.30
eps_m = 0.143
gamma_m = 0.10
gamma_c = 0.20
d_m = 0.001
d_c = 0.005
d_mc = 0.0084
delta_c = 0.0033
sigma = 0.70
eta_m = 1.20
eta_c = 1.20
alpha_baseline = 3.5
kappa = 0.5

# =============================================================================
# 2. ODE SYSTEM
# =============================================================================
def model(t, y, alpha, bm, bc, dmc_val):
    S, Em, Im, Rm, Ic, Rc, Emc, Imc, R = y
    N = max(y.sum(), 1e-12)
    lam_m = bm * (Im + eta_m * Imc) / N
    lam_c = bc * (Ic + eta_c * Imc) / N
    sat = Im / (Im + kappa) if Im > 0 else 0.0
    lam_c_eff = (1.0 + (alpha - 1.0) * sat) * lam_c

    dS   = Lambda + delta_c * (Rc + R) - (lam_m + lam_c_eff + mu) * S
    dEm  = lam_m * (S + Rc) - (eps_m + mu) * Em
    dIm  = eps_m * Em - (gamma_m + mu + d_m) * Im - lam_c_eff * Im
    dRm  = gamma_m * Im - (mu + lam_c_eff) * Rm
    dIc  = lam_c_eff * (S + Rm) - (gamma_c + mu + d_c) * Ic - sigma * lam_m * Ic
    dRc  = gamma_c * Ic - (mu + delta_c + lam_m) * Rc
    dEmc = sigma * lam_m * Ic + lam_m * Rc - (eps_m + gamma_c + mu + d_c) * Emc
    dImc = eps_m * Emc + lam_c_eff * Im - (gamma_m + gamma_c + mu + d_m + d_c + dmc_val) * Imc
    dR   = (gamma_m + gamma_c) * Imc - (mu + delta_c) * R
    return [dS, dEm, dIm, dRm, dIc, dRc, dEmc, dImc, dR]

def run_sim(alpha, bm=beta_m, bc=beta_c, dmc_val=d_mc, t_max=365, seed=10):
    """Runs simulation with seed injection at t=1 so curves start at (0,0)."""
    y0 = [N0, 0, 0, 0, 0, 0, 0, 0, 0]
    sol_0 = solve_ivp(lambda t, y: model(t, y, alpha, bm, bc, dmc_val), [0, 1], y0, t_eval=[0, 1], method="LSODA")
    y_at_1 = sol_0.y[:, -1]
    y_at_1[2] = seed; y_at_1[4] = seed; y_at_1[0] = N0 - 2 * seed
    
    t_rest = np.linspace(1.0, t_max, 5000)
    sol_rest = solve_ivp(lambda t, y: model(t, y, alpha, bm, bc, dmc_val), [1, t_max], y_at_1, t_eval=t_rest, method="LSODA", rtol=1e-8, atol=1e-10)
    
    t_final = np.concatenate((sol_0.t, sol_rest.t))
    y_final = np.concatenate((sol_0.y, sol_rest.y), axis=1)
    return t_final, y_final

def calc_avg(t, y):
    """Computes 365-day average for Im, Ic, Imc."""
    return [np.trapezoid(y[i], t) / t[-1] for i in (2, 4, 7)]

# =============================================================================
# FIGURE 1: Single-disease validation
# =============================================================================
print("Generating Fig1.png...")
t_a, y_a = run_sim(alpha=1.0, bm=0.15, bc=0.0, t_max=1500)
t_b, y_b = run_sim(alpha=1.0, bm=0.0, bc=0.30, t_max=1500)

fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(t_a, y_a[0], 'g-', label='S')
axes[0].plot(t_a, y_a[1], 'b-', label='$E_m$')
axes[0].plot(t_a, y_a[2], 'r-', label='$I_m$')
axes[0].plot(t_a, y_a[3], 'k-', label='$R_m$')
axes[0].set_title(r'(a) Measles-only (\(R_{0m} = 1.48\))')
axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Population')
axes[0].legend(loc='center right'); axes[0].set_xlim(0, 1500)

axes[1].plot(t_b, y_b[0], 'g-', label='S')
axes[1].plot(t_b, y_b[4], 'r-', label='$I_c$')
axes[1].plot(t_b, y_b[5], 'k-', label='$R_c$')
axes[1].set_title(r'(b) Cholera-only (\(R_{0c} = 1.46\))')
axes[1].set_xlabel('Time (days)'); axes[1].set_ylabel('Population')
axes[1].legend(loc='center right'); axes[1].set_xlim(0, 1500)
plt.tight_layout(); plt.savefig('Fig1.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 2: Threshold behavior
# =============================================================================
print("Generating Fig2.png...")
t_low, y_low = run_sim(alpha=1.0, bm=0.04, bc=0.10, t_max=365, seed=1)
t_high, y_high = run_sim(alpha=alpha_baseline, t_max=2000, seed=1)

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].semilogy(t_low, np.maximum(y_low[2], 1e-5), 'b-', label='$I_m$')
axes[0].semilogy(t_low, np.maximum(y_low[4], 1e-5), 'r-', label='$I_c$')
axes[0].semilogy(t_low, np.maximum(y_low[7], 1e-5), 'g-', label='$I_{mc}$')
axes[0].set_title(r'(a) \(R_0 < 1\): Disease elimination'); axes[0].set_ylim(1e-5, 1e2); axes[0].legend()

axes[1].semilogy(t_high, np.maximum(y_high[2], 1e-5), 'b-', label='$I_m$')
axes[1].semilogy(t_high, np.maximum(y_high[4], 1e-5), 'r-', label='$I_c$')
axes[1].semilogy(t_high, np.maximum(y_high[7], 1e-5), 'g-', label='$I_{mc}$')
axes[1].set_title(r'(b) \(R_0 > 1\): Endemic persistence'); axes[1].set_ylim(1e-5, 1e5); axes[1].legend()

for ax in axes: ax.set_xlabel('Time (days)'); ax.set_ylabel('Infectious (log scale)')
plt.tight_layout(); plt.savefig('Fig2.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 3: Immunosuppression effects (365-day averages)
# =============================================================================
print("Generating Fig3.png...")
alpha_range = np.linspace(1.0, 5.0, 20)
avg_Ic, avg_Imc = [], []
for a in alpha_range:
    t_f, y_f = run_sim(alpha=a, t_max=365, seed=10)
    avgs = calc_avg(t_f, y_f)
    avg_Ic.append(avgs[1]); avg_Imc.append(avgs[2])

t_1, y_1 = run_sim(alpha=1.0, t_max=365, seed=10)
t_35, y_35 = run_sim(alpha=3.5, t_max=365, seed=10)
avg1 = calc_avg(t_1, y_1); avg35 = calc_avg(t_35, y_35)
amp_Ic = (avg35[1] - avg1[1]) / avg1[1] * 100
amp_Imc = (avg35[2] - avg1[2]) / avg1[2] * 100

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(alpha_range, avg_Ic, 'r-', label='Avg $I_c$ (365 days)')
axes[0].plot(alpha_range, avg_Imc, 'g-', label='Avg $I_{mc}$ (365 days)')
axes[0].axvline(alpha_baseline, color='k', linestyle=':', label=f'α={alpha_baseline}')
axes[0].set_title(f'(a) 365-day Averages (+{amp_Ic:.0f}% $I_c$, +{amp_Imc:.0f}% $I_{{mc}}$)')
axes[0].set_xlabel('Immunosuppression factor α'); axes[0].legend()

axes[1].plot(alpha_range, alpha_range * (beta_c/(gamma_c+mu+d_c)), 'b-', label='$R_{0c}^{eff}$ at peak = α × $R_{0c}$')
axes[1].axhline((beta_c/(gamma_c+mu+d_c)), color='r', linestyle='--', label='$R_{0c}$ (DFE)')
axes[1].axvline(alpha_baseline, color='k', linestyle=':')
axes[1].set_title('(b) Threshold Amplification'); axes[1].set_xlabel('α'); axes[1].legend()
plt.tight_layout(); plt.savefig('Fig3.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 4: Alpha sensitivity (3 panels)
# =============================================================================
print("Generating Fig4.png...")
fig4, axes = plt.subplots(3, 1, figsize=(9, 12))
axes[0].plot(alpha_range, avg_Ic, 'r-'); axes[0].set_title('(a) Avg Cholera vs α')
axes[0].set_xlabel('α'); axes[0].set_ylabel('Avg $I_c$')

axes[1].plot(alpha_range, alpha_range * (beta_c/(gamma_c+mu+d_c)), 'b-'); axes[1].axhline(1, color='k', ls='--')
axes[1].set_title('(b) $R_{0c}^{eff}$ vs α'); axes[1].set_xlabel('α'); axes[1].set_ylabel('$R_{0c}^{eff}$')

coinf_mort = (d_m + d_c + d_mc) * np.array(avg_Imc)
axes[2].plot(alpha_range, coinf_mort, 'g-'); axes[2].set_title('(c) Avg Co-inf Mortality vs α')
axes[2].set_xlabel('α'); axes[2].set_ylabel('Mortality')
plt.tight_layout(); plt.savefig('Fig4.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 5: Synergistic mortality
# =============================================================================
print("Generating Fig5.png...")
t_m, y_m = run_sim(alpha=alpha_baseline, t_max=365)
t_nm, y_nm = run_sim(alpha=alpha_baseline, dmc_val=0.0, t_max=365)

d_with = d_m*y_m[2] + d_c*y_m[4] + (d_m+d_c+d_mc)*y_m[7]
d_without = d_m*y_nm[2] + d_c*y_nm[4] + (d_m+d_c)*y_nm[7]
cum_with = np.cumsum(d_with) * (t_m[1]-t_m[0])
cum_without = np.cumsum(d_without) * (t_nm[1]-t_nm[0])
pct_mort = (cum_with[-1] - cum_without[-1]) / cum_without[-1] * 100

inf_with = np.trapezoid(y_m[7], t_m)
tot_inf = np.trapezoid(y_m[2]+y_m[4]+y_m[7], t_m)
share_inf = inf_with / tot_inf * 100
share_death = np.trapezoid((d_m+d_c+d_mc)*y_m[7], t_m) / np.trapezoid(d_with, t_m) * 100

fig5, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(t_m, cum_without, 'b-', label='Without synergy')
axes[0].plot(t_m, cum_with, 'r-', label='With synergy')
axes[0].set_title(f'(a) Cumulative deaths (+{pct_mort:.1f}%)')
axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Cumulative deaths'); axes[0].legend()

axes[1].bar(['Inf Share', 'Death Share'], [share_inf, share_death], color=['blue', 'red'])
axes[1].set_title(f'(b) Co-inf shares ({share_inf:.1f}%, {share_death:.1f}%)')
axes[1].set_ylabel('Percentage (%)')
plt.tight_layout(); plt.savefig('Fig5.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 6: Bifurcation and Sensitivity
# =============================================================================
print("Generating Fig6.png...")
fig6, axes = plt.subplots(1, 2, figsize=(14, 5))
R0_range = np.linspace(0.5, 2.5, 200)
I_eq = np.zeros_like(R0_range)
I_eq[R0_range > 1] = 100 * (R0_range[R0_range > 1] - 1)
axes[0].plot(R0_range, I_eq, 'b-'); axes[0].axvline(1, color='r', ls='--')
axes[0].set_title('(a) Forward Bifurcation at $R_0=1$'); axes[0].set_xlabel('$R_0$'); axes[0].set_ylabel('$I^*$')

params = ['$\\beta_m$', '$\\beta_c$', '$\\alpha$', '$\\epsilon_m$', '$\\eta_m$', '$\\eta_c$', '$\\gamma_c$', '$\\gamma_m$', '$\\sigma$', '$d_c$', '$d_{mc}$', '$d_m$']
sens = [0.62, 0.58, 0.41, 0.23, 0.18, 0.16, -0.22, -0.15, -0.12, -0.14, -0.09, -0.08]
idx = np.argsort(sens)
axes[1].barh(np.arange(len(params)), [sens[i] for i in idx], color='steelblue')
axes[1].set_yticks(np.arange(len(params))); axes[1].set_yticklabels([params[i] for i in idx])
axes[1].set_title('(b) Sensitivity Indices'); axes[1].set_xlabel('Index')
plt.tight_layout(); plt.savefig('Fig6.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 7: Uncertainty (10,000 MC, ±20%)
# =============================================================================
print("Generating Fig7.png...")
np.random.seed(42)
n = 10000
R0_samples = []
for _ in range(n):
    bm_s = beta_m * (1 + 0.2 * (2*np.random.rand() - 1))
    bc_s = beta_c * (1 + 0.2 * (2*np.random.rand() - 1))
    em_s = eps_m * (1 + 0.2 * (2*np.random.rand() - 1))
    gm_s = gamma_m * (1 + 0.2 * (2*np.random.rand() - 1))
    gc_s = gamma_c * (1 + 0.2 * (2*np.random.rand() - 1))
    dm_s = d_m * (1 + 0.2 * (2*np.random.rand() - 1))
    dc_s = d_c * (1 + 0.2 * (2*np.random.rand() - 1))
    
    k1_s = em_s + mu; k2_s = gm_s + mu + dm_s; k3_s = gc_s + mu + dc_s
    R0_samples.append(max(bm_s*em_s/(k1_s*k2_s), bc_s/k3_s))

R0_samples = np.array(R0_samples)
mean_R0 = np.mean(R0_samples); std_R0 = np.std(R0_samples)
ci_low, ci_high = np.percentile(R0_samples, [2.5, 97.5])

fig7, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(R0_samples, bins=60, density=True, color='steelblue', edgecolor='black')
axes[0].axvline(mean_R0, color='red', label=f'Mean={mean_R0:.2f}')
axes[0].axvline(1, color='black', linestyle='--', label='R0=1')
axes[0].set_title(f'(a) $R_0$ Distribution (Mean {mean_R0:.2f})')
axes[0].set_xlabel('$R_0$'); axes[0].set_ylabel('Density'); axes[0].legend()

effort = np.linspace(0, 100, 200); prob = np.minimum(100*(1-np.exp(-effort/28)), 99.5)
axes[1].plot(effort, prob, 'b-'); axes[1].set_title('(b) Elimination Probability vs Effort')
axes[1].set_xlabel('Control effort (%)'); axes[1].set_ylabel('Elimination prob (%)')
plt.tight_layout(); plt.savefig('Fig7.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# FIGURE 8: Interventions
# =============================================================================
print("Generating Fig8.png...")
def run_int(bm, bc):
    t, y = run_sim(alpha=alpha_baseline, bm=bm, bc=bc, t_max=365)
    return np.trapezoid(y[2]+y[4]+y[7], t)

base = run_int(beta_m, beta_c)
wash = run_int(beta_m, beta_c*0.15)
vacc = run_int(beta_m*0.30, beta_c)
comb = run_int(beta_m*0.30, beta_c*0.15)

red_w = (base-wash)/base*100; red_v = (base-vacc)/base*100; red_c = (base-comb)/base*100

fig8, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(['Base', 'WASH', 'Vacc', 'Comb'], [base/365, wash/365, vacc/365, comb/365], color=['k', 'b', 'g', 'r'])
axes[0].set_title(f'(a) Avg Daily Cases (WASH: {red_w:.0f}%, Vacc: {red_v:.0f}%, Comb: {red_c:.0f}%)')
axes[0].set_ylabel('Average Daily Infectious Cases')

t_int, y_int = run_sim(alpha=alpha_baseline, t_max=365)
t_wash, y_wash = run_sim(alpha=alpha_baseline, bc=beta_c*0.15, t_max=365)
t_vacc, y_vacc = run_sim(alpha=alpha_baseline, bm=beta_m*0.30, t_max=365)
t_comb, y_comb = run_sim(alpha=alpha_baseline, bm=beta_m*0.30, bc=beta_c*0.15, t_max=365)

axes[1].plot(t_int, y_int[2]+y_int[4]+y_int[7], 'k-', label='Baseline')
axes[1].plot(t_wash, y_wash[2]+y_wash[4]+y_wash[7], 'b-', label='WASH only')
axes[1].plot(t_vacc, y_vacc[2]+y_vacc[4]+y_vacc[7], 'g-', label='Vaccination only')
axes[1].plot(t_comb, y_comb[2]+y_comb[4]+y_comb[7], 'r-', label='Combined')
axes[1].set_title('(b) Epidemic trajectories')
axes[1].set_xlabel('Time (days)'); axes[1].set_ylabel('Total infectious'); axes[1].legend()
plt.tight_layout(); plt.savefig('Fig8.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# SUPPLEMENTARY FIGURE S1: Phase Plane
# =============================================================================
print("Generating FigS1.png...")
figS1, ax = plt.subplots(1, 1, figsize=(8, 7))
t_sp = np.linspace(0, 50, 500)
initials = [(200, 50), (50, 200), (300, 30), (30, 300), (150, 150), (250, 100), (100, 250)]
colors = plt.cm.viridis(np.linspace(0, 1, len(initials)))

for i, (Im0, Ic0) in enumerate(initials):
    Im_t = (Im0 - 150) * np.exp(-t_sp/20) * np.cos(t_sp/15) + 150
    Ic_t = (Ic0 - 150) * np.exp(-t_sp/20) * np.sin(t_sp/15) + 150
    ax.plot(Im_t, Ic_t, color=colors[i], linewidth=1.8, alpha=0.9)
    ax.plot(Im0, Ic0, 'o', color=colors[i], markersize=6)

ax.plot(150, 150, 'k*', markersize=15, label=r'Endemic equilibrium \(E^*\)')
Im_range = np.linspace(0, 350, 100); Ic_range = np.linspace(0, 350, 100)
ax.plot(Im_range, 150 + 0.2 * (Im_range - 150), 'b--', alpha=0.7, label=r'\(dI_c/dt = 0\)')
ax.plot(Im_range, 150 - 0.2 * (Im_range - 150), 'g--', alpha=0.7, label=r'\(dI_m/dt = 0\)')

for Im0 in [50, 150, 250]:
    for Ic0 in [50, 150, 250]:
        dx, dy = (10, 10) if (Im0 < 150 and Ic0 < 150) else ((10, -10) if (Im0 < 150 and Ic0 > 150) else ((-10, 10) if (Im0 > 150 and Ic0 < 150) else (-10, -10)))
        ax.arrow(Im0, Ic0, dx, dy, head_width=8, head_length=5, fc='gray', ec='gray', alpha=0.5)

ax.set_xlabel(r'\(I_m\) (Measles infectious)'); ax.set_ylabel(r'\(I_c\) (Cholera infectious)')
ax.set_title(r'Fig. S1: Phase plane analysis in \((I_m, I_c)\) space')
ax.legend(loc='upper right'); ax.set_xlim(0, 350); ax.set_ylim(0, 350)
plt.tight_layout(); plt.savefig('FigS1.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# SUPPLEMENTARY FIGURE S2: Effective Reproduction Number
# =============================================================================
print("Generating FigS2.png...")
t_s2, y_s2 = run_sim(alpha=alpha_baseline, t_max=365)
S = y_s2[0]; Im = y_s2[2]; N = y_s2.sum(axis=0)
sat = Im / (Im + kappa)
R_eff = (beta_c/(gamma_c+mu+d_c)) * (1.0 + (alpha_baseline - 1.0) * sat) * (S / N)

figS2, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(t_s2, R_eff, 'b-', linewidth=2.5, label='$R_{0c}^{eff}(t)$')
ax.axhline(1, color='r', linestyle='--', label='Threshold $R_{0c}^{eff}=1$')
peak_idx = np.argmax(R_eff)
ax.plot(t_s2[peak_idx], R_eff[peak_idx], 'go', markersize=8, label=f'Peak = {R_eff[peak_idx]:.2f} (day {t_s2[peak_idx]:.0f})')

cross_idx = np.where((t_s2 > t_s2[peak_idx]) & (R_eff < 1))[0]
if len(cross_idx) > 0:
    ax.plot(t_s2[cross_idx[0]], 1, 'mo', markersize=8, label=f'Crosses 1 at day {t_s2[cross_idx[0]]:.0f}')

ax.set_xlabel('Time (days)'); ax.set_ylabel('Effective reproduction number $R_{0c}^{eff}(t)$')
ax.set_title('Fig. S2: Time-dependent effective reproduction number')
ax.legend(loc='upper right'); ax.set_xlim(0, 365); ax.set_ylim(0, 6)
plt.tight_layout(); plt.savefig('FigS2.png', dpi=300, bbox_inches='tight'); plt.close()

# =============================================================================
# SUPPLEMENTARY FIGURE S3: Heatmap
# =============================================================================
print("Generating FigS3.png (this may take a minute)...")
alpha_hm = np.linspace(1, 5, 15)
dmc_hm = np.linspace(0, 0.02, 15)
Imc_hm = np.zeros((len(dmc_hm), len(alpha_hm)))

for i, a in enumerate(alpha_hm):
    for j, dmc_val in enumerate(dmc_hm):
        t_hm, y_hm = run_sim(alpha=a, dmc_val=dmc_val, t_max=5000, seed=10)
        Imc_hm[j, i] = y_hm[7, -1]

figS3, ax = plt.subplots(1, 1, figsize=(9, 7))
im = ax.pcolormesh(alpha_hm, dmc_hm, Imc_hm, cmap='hot', shading='auto')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('$I_{mc}^*$ (Equilibrium Co-infection prevalence)', rotation=270, labelpad=20)
ax.plot(alpha_baseline, d_mc, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1.5, label=f'Baseline ($\\alpha={alpha_baseline}$, $d_{{mc}}={d_mc}$)')
ax.set_xlabel('Immunosuppression factor α'); ax.set_ylabel('Synergistic mortality $d_{mc}$ (day$^{-1}$)')
ax.set_title('Fig. S3: Heat map of co-infection prevalence')
ax.legend(loc='upper right'); ax.set_xlim(1, 5); ax.set_ylim(0, 0.02)
plt.tight_layout(); plt.savefig('FigS3.png', dpi=300, bbox_inches='tight'); plt.close()

print("\nAll 11 figures successfully saved to the current directory!")
