# Validation R pour régression robuste, polynomiale et splines

df <- read.csv("validation/data_robust_python.csv")

# ============================================================================
# 1. Régression Robuste (MASS::rlm)
# ============================================================================
cat("=== RÉGRESSION ROBUSTE (Huber) ===\n")
library(MASS)
fit_robust <- rlm(y ~ x, data = df, psi = psi.huber)
print(summary(fit_robust))
cat("\nCoefficients:\n")
print(coef(fit_robust))

# ============================================================================
# 2. Régression Polynomiale
# ============================================================================
cat("\n=== RÉGRESSION POLYNOMIALE (degré 2, raw=TRUE) ===\n")
fit_poly <- lm(y ~ poly(x, 2, raw = TRUE), data = df)
print(summary(fit_poly))
cat("\nCoefficients:\n")
print(coef(fit_poly))

# ============================================================================
# 3. Régression par Splines (splines::bs)
# ============================================================================
cat("\n=== RÉGRESSION PAR SPLINES (splines::bs) ===\n")
library(splines)
# bs() accepte df et degree, comme patsy.bs()
fit_spline <- lm(y ~ bs(x, df = 4, degree = 3), data = df)
print(summary(fit_spline))
cat("\nCoefficients:\n")
print(coef(fit_spline))
cat("\nR²:", summary(fit_spline)$r.squared, "\n")
cat("\nR² ajusté:", summary(fit_spline)$adj.r.squared, "\n")