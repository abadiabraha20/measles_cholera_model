# Measles-Cholera Co-infection Model

This repository contains the complete Python code for the mathematical model presented in:

**"Quantifying Synergy in Measles-Cholera Co-infection: A Mathematical Model with Immunosuppression"**

*Asgedom A.A. and Welua T.A.*

---

## 📖 Overview

This code implements a nine-compartment deterministic model for measles-cholera co-infection with immunosuppression. The model quantifies how measles-induced immunosuppression ($\alpha$) amplifies cholera transmission and mortality.

### Key Features:
- **9 compartments**: S, E_m, I_m, R_m, I_c, R_c, E_mc, I_mc, R
- **Immunosuppression parameter** $\alpha$ (2-4 fold increase in susceptibility)
- **Synergistic mortality** $d_{mc}$ (40% increase)
- **Complete analytical derivation** of $\mathcal{R}_0$
- **Forward bifurcation** analysis at $\mathcal{R}_0=1$
- **Sensitivity analysis** and uncertainty quantification
- **Intervention strategies** (WASH, vaccination, combined)

---

## 📁 Repository Structure
measles_cholera_model/
│
├── generate_figures.py # Main script to generate all figures
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── figures/ # Main figures (1-8)
│ ├── Fig1_single_disease_a.png
│ ├── Fig1_single_disease_b.png
│ ├── Fig2_threshold_a.png
│ ├── Fig2_threshold_b.png
│ ├── Fig3_immunosuppression_a.png
│ ├── Fig3_immunosuppression_b.png
│ ├── Fig4_alpha_sensitivity.png
│ ├── Fig5_mortality_a.png
│ ├── Fig5_mortality_b.png
│ ├── Fig6_bifurcation_a.png
│ ├── Fig6_bifurcation_b.png
│ ├── Fig7_uncertainty_a.png
│ ├── Fig7_uncertainty_b.png
│ ├── Fig8_interventions_a.png
│ └── Fig8_interventions_b.png
│
└── supplementary/ # Supplementary figures
├── Fig_S1_phase_plane.png
├── Fig_S2_effective_R0.png
└── Fig_S3_heatmap.png
---
