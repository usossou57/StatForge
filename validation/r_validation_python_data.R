# Validation R sur les données générées par Python

# ============================================================================
# 1. Données binaires (Probit)
# ============================================================================
cat("=== PROBIT (données Python) ===\n")
df_binary <- read.csv("validation/data_binary_python.csv")
fit_probit <- glm(y ~ x1 + x2, data = df_binary, family = binomial(link = "probit"))
print(summary(fit_probit))
cat("\nCoefficients:\n")
print(coef(fit_probit))

# ============================================================================
# 2. Données de comptage (Poisson et Binomiale Négative)
# ============================================================================
cat("\n=== POISSON (données Python) ===\n")
df_count <- read.csv("validation/data_count_python.csv")
fit_poisson <- glm(y ~ x1 + x2, data = df_count, family = poisson)
print(summary(fit_poisson))
cat("\nCoefficients:\n")
print(coef(fit_poisson))

cat("\n=== BINOMIALE NÉGATIVE (données Python) ===\n")
library(MASS)
fit_nb <- glm.nb(y ~ x1 + x2, data = df_count)
print(summary(fit_nb))
cat("\nCoefficients:\n")
print(coef(fit_nb))
cat("\nTheta (1/alpha):", fit_nb$theta, "\n")

# ============================================================================
# 3. Données positives (Gamma)
# ============================================================================
cat("\n=== GAMMA (données Python) ===\n")
df_positive <- read.csv("validation/data_positive_python.csv")
fit_gamma <- glm(y ~ x1 + x2, data = df_positive, family = Gamma(link = "log"))
print(summary(fit_gamma))
cat("\nCoefficients:\n")
print(coef(fit_gamma))