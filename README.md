# Measles-Cholera Co-infection Model

This repository contains the complete, scientifically rigorous Python code for the mathematical model presented in:

*"Quantifying Synergy in Measles-Cholera Co-infection: A Mathematical Model with Immunosuppression"*

*Asgedom A.A. and Welu T.A.*

---

## 📖 Overview

This code implements a nine-compartment deterministic ODE model for measles-cholera co-infection. The model explicitly quantifies how measles-induced immunosuppression ($\alpha$) amplifies cholera transmission and synergistic mortality ($d_{mc}$) during concurrent outbreaks. 

All figures in the manuscript and supplementary material are generated directly from numerical integration of the ODE system using `scipy.integrate.solve_ivp`, ensuring mathematical accuracy and reproducibility.

### Key Features:
- **9 compartments**: S, E_m, I_m, R_m, I_c, R_c, E_mc, I_mc, R
- **Immunosuppression parameter** $\alpha = 3.5$ (conditional amplification of cholera susceptibility during measles infection)
- **Synergistic mortality** $d_{mc} = 0.0084$ (40% increase in co-infection mortality)
- **Complete analytical derivation** of the basic reproduction number $\mathcal{R}_0 = \max\{\mathcal{R}_{0m}, \mathcal{R}_{0c}\}$
- **Forward transcritical bifurcation** at $\mathcal{R}_0 = 1$
- **Sensitivity analysis** (normalized indices) and Monte Carlo uncertainty quantification (10,000 samples)
- **Intervention strategies** (WASH, vaccination, and combined interventions)
- **Transient dynamics**: The code demonstrates that synergy is a transient phenomenon, with a **+122%** average cholera amplification and **+192%** average co-infection amplification over a 365-day window, peaking at an effective reproduction number of **5.12**.

---

## 📁 Repository Structure

```text
measles_cholera_model/
│
├── full_manuscript_figures.py   # Main script to generate all figures
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── Fig1.png                     # Single-disease validation
├── Fig2.png                     # Threshold behavior
├── Fig3.png                     # Immunosuppression effects
├── Fig4.png                     # Alpha sensitivity analysis
├── Fig5.png                     # Synergistic mortality
├── Fig6.png                     # Bifurcation and sensitivity
├── Fig7.png                     # Uncertainty and robustness
├── Fig8.png                     # Intervention strategies
├── FigS1.png                    # Phase plane analysis
├── FigS2.png                    # Effective reproduction number
└── FigS3.png                    # Heatmap of co-infection prevalence
📜 License & Citation
This code is provided for reproducibility and peer-review validation. If you use this code or model in your research, please cite:

Asgedom, A. A., & Welu, T. A. (2026). Quantifying Synergy in Measles-Cholera Co-infection: A Mathematical Model with Immunosuppression. Journal of Applied Mathematics.
