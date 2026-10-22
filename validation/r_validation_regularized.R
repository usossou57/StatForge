# Validation R pour régression régularisée

# Charger les données générées par Python
df <- read.csv("validation/data_regularized_python.csv")

# ============================================================================
# 1. Ridge (MASS::lm.ridge)
# ============================================================================
cat("=== RIDGE (MASS::lm.ridge) ===\n")
library(MASS)
fit_ridge <- lm.ridge(y ~ x1 + x2 + x3, data = df, lambda = 1.0)
print(fit_ridge)
cat("\nCoefficients:\n")
print(coef(fit_ridge))

# ============================================================================
# 2. Lasso (glmnet)
# ============================================================================
cat("\n=== LASSO (glmnet) ===\n")
library(glmnet)
X <- as.matrix(df[, c("x1", "x2", "x3")])
y <- df$y

# Standardiser
X_scaled <- scale(X)

# Ajuster avec lambda = 0.1 (équivalent alpha=0.1 dans sklearn)
# Note: glmnet utilise lambda, pas alpha
fit_lasso <- glmnet(X_scaled, y, alpha = 1.0, lambda = 0.1, standardize = FALSE)
print(fit_lasso)
cat("\nCoefficients:\n")
print(coef(fit_lasso, s = 0.1))

# ============================================================================
# 3. Elastic Net (glmnet)
# ============================================================================
cat("\n=== ELASTIC NET (glmnet) ===\n")
fit_enet <- glmnet(X_scaled, y, alpha = 0.5, lambda = 0.1, standardize = FALSE)
print(fit_enet)
cat("\nCoefficients:\n")
print(coef(fit_enet, s = 0.1))