"""
Gabarit reproductible pour GLM (probit, Poisson, binomiale négative, Gamma).
"""

import pandas as pd
from app.statistics.regression_glm import (
    fit_probit,
    fit_poisson,
    fit_negative_binomial,
    fit_gamma,
)

# Charger les données
df = pd.read_csv("data/example.csv")

# ============================================================================
# 1. Probit (données binaires)
# ============================================================================
result_probit = fit_probit(
    df=df,
    formula="y_binary ~ x1 + x2 + x3",
    link="probit",
)
print("=== PROBIT ===")
print(result_probit["summary"])

# ============================================================================
# 2. Poisson (comptages)
# ============================================================================
result_poisson = fit_poisson(
    df=df,
    formula="y_count ~ x1 + x2 + x3",
    link="log",
)
print("\n=== POISSON ===")
print(result_poisson["summary"])

# ============================================================================
# 3. Binomiale Négative (comptages surdispersés)
# ============================================================================
result_nb = fit_negative_binomial(
    df=df,
    formula="y_count ~ x1 + x2 + x3",
    link="log",
    log_alpha=0.0,
)
print("\n=== BINOMIALE NÉGATIVE ===")
print(result_nb["summary"])
print(f"Alpha (dispersion): {result_nb['alpha']:.4f}")

# ============================================================================
# 4. Gamma (continues positives)
# ============================================================================
result_gamma = fit_gamma(
    df=df,
    formula="y_positive ~ x1 + x2 + x3",
    link="log",
)
print("\n=== GAMMA ===")
print(result_gamma["summary"])