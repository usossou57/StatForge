# Validation R pour régression ordinale

# Charger les données générées par Python
df_ordinal <- read.csv("validation/data_ordinal_python.csv")

# Convertir y en facteur ordonné
df_ordinal$y <- factor(df_ordinal$y, levels = c(1, 2, 3, 4), ordered = TRUE)

# ============================================================================
# Régression ordinale avec MASS::polr (logit)
# ============================================================================
cat("=== RÉGRESSION ORDINALE (logit) ===\n")
library(MASS)
fit_ordinal_logit <- polr(y ~ x1 + x2, data = df_ordinal, method = "logistic", Hess = TRUE)
print(summary(fit_ordinal_logit))
cat("\nCoefficients:\n")
print(coef(fit_ordinal_logit))
cat("\nIntercepts (thresholds):\n")
print(fit_ordinal_logit$zeta)

# ============================================================================
# Régression ordinale avec MASS::polr (probit)
# ============================================================================
cat("\n=== RÉGRESSION ORDINALE (probit) ===\n")
fit_ordinal_probit <- polr(y ~ x1 + x2, data = df_ordinal, method = "probit", Hess = TRUE)
print(summary(fit_ordinal_probit))
cat("\nCoefficients:\n")
print(coef(fit_ordinal_probit))
cat("\nIntercepts (thresholds):\n")
print(fit_ordinal_probit$zeta)