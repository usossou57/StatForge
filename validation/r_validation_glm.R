# Validation R pour GLM (probit, Poisson, binomiale négative, Gamma)

# Générer les mêmes données que Python
set.seed(42)
n <- 200

# Données binaires
x1 <- rnorm(n)
x2 <- rnorm(n)
prob <- 1 / (1 + exp(-(0.5 * x1 + 0.3 * x2)))
y_binary <- rbinom(n, 1, prob)
df_binary <- data.frame(y = y_binary, x1 = x1, x2 = x2)

# Données de comptage
lam <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
y_count <- rpois(n, lam)
df_count <- data.frame(y = y_count, x1 = x1, x2 = x2)

# Données continues positives
scale <- exp(1 + 0.3 * x1 + 0.2 * x2)
y_positive <- rgamma(n, shape = 2, scale = scale / 2)
df_positive <- data.frame(y = y_positive, x1 = x1, x2 = x2)

# ============================================================================
# 1. Probit
# ============================================================================
cat("=== PROBIT ===\n")
fit_probit <- glm(y ~ x1 + x2, data = df_binary, family = binomial(link = "probit"))
print(summary(fit_probit))
cat("\nCoefficients:\n")
print(coef(fit_probit))

# ============================================================================
# 2. Poisson
# ============================================================================
cat("\n=== POISSON ===\n")
fit_poisson <- glm(y ~ x1 + x2, data = df_count, family = poisson)
print(summary(fit_poisson))
cat("\nCoefficients:\n")
print(coef(fit_poisson))

# ============================================================================
# 3. Binomiale Négative
# ============================================================================
cat("\n=== BINOMIALE NÉGATIVE ===\n")
library(MASS)
fit_nb <- glm.nb(y ~ x1 + x2, data = df_count)
print(summary(fit_nb))
cat("\nCoefficients:\n")
print(coef(fit_nb))
cat("\nTheta (1/alpha):", fit_nb$theta, "\n")

# ============================================================================
# 4. Gamma
# ============================================================================
cat("\n=== GAMMA ===\n")
fit_gamma <- glm(y ~ x1 + x2, data = df_positive, family = Gamma(link = "log"))
print(summary(fit_gamma))
cat("\nCoefficients:\n")
print(coef(fit_gamma))