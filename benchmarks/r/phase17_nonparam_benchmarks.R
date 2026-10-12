# ============================================================================
# Benchmarks R pour validation croisée (Phase 17 — Tests non paramétriques)
# ============================================================================
# Ce script calcule les mêmes statistiques que les fonctions Python
# pour validation croisée. Les valeurs attendues sont utilisées dans
# tests/test_statistics/test_tests_nonparam.py (TestRBenchmarkValidation).
#
# Exécution : Rscript benchmarks/r/phase17_nonparam_benchmarks.R
# ============================================================================

cat("=== Benchmarks R pour Phase 17 — Tests non paramétriques ===\n\n")

# ----------------------------------------------------------------------------
# 1. Test de Mann-Whitney U
# ----------------------------------------------------------------------------
cat("1. Test de Mann-Whitney U\n")
group_a <- c(10.2, 11.5, 9.8, 10.9, 11.2, 10.5, 11.8, 10.1, 10.7, 11.0)
group_b <- c(12.5, 13.1, 11.8, 12.9, 13.5, 12.2, 13.8, 12.1, 12.7, 13.0)
df <- data.frame(
  outcome = c(group_a, group_b),
  group = factor(c(rep("A", 10), rep("B", 10)))
)

result <- wilcox.test(outcome ~ group, data = df, exact = FALSE, correct = TRUE)
cat(sprintf("   W = %.4f\n", result$statistic))
cat(sprintf("   p = %.6f\n", result$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 2. Test de Wilcoxon signed-rank
# ----------------------------------------------------------------------------
cat("2. Test de Wilcoxon signed-rank\n")
before <- c(10.2, 11.5, 9.8, 10.9, 11.2, 10.5, 11.8, 10.1, 10.7, 11.0)
after <- c(11.5, 12.1, 10.5, 11.8, 12.0, 11.2, 12.5, 10.9, 11.4, 11.7)

result <- wilcox.test(before, after, paired = TRUE, exact = FALSE, correct = TRUE)
cat(sprintf("   V = %.4f\n", result$statistic))
cat(sprintf("   p = %.6f\n", result$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 3. Test de Kruskal-Wallis
# ----------------------------------------------------------------------------
cat("3. Test de Kruskal-Wallis\n")
group_a <- c(10.2, 11.5, 9.8, 10.9, 11.2)
group_b <- c(12.5, 13.1, 11.8, 12.9, 13.5)
group_c <- c(14.5, 15.1, 13.8, 14.9, 15.5)
df <- data.frame(
  outcome = c(group_a, group_b, group_c),
  group = factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))
)

result <- kruskal.test(outcome ~ group, data = df)
cat(sprintf("   Chi² = %.4f\n", result$statistic))
cat(sprintf("   df = %.1f\n", result$parameter))
cat(sprintf("   p = %.6f\n", result$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 4. Test de Friedman
# ----------------------------------------------------------------------------
cat("4. Test de Friedman\n")
set.seed(42)
n <- 20
cond1 <- rnorm(n, 10, 2)
cond2 <- cond1 + rnorm(n, 0.5, 0.5)
cond3 <- cond1 + rnorm(n, 1.0, 0.5)
df <- data.frame(cond1 = cond1, cond2 = cond2, cond3 = cond3)

# Friedman test nécessite une matrice
mat <- as.matrix(df)
result <- friedman.test(mat)
cat(sprintf("   Chi² = %.4f\n", result$statistic))
cat(sprintf("   df = %.1f\n", result$parameter))
cat(sprintf("   p = %.6f\n", result$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 5. Test de Spearman
# ----------------------------------------------------------------------------
cat("5. Test de corrélation de Spearman\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2, 4, 5, 4, 5, 6, 7, 8, 9, 10)

result <- cor.test(x, y, method = "spearman", exact = FALSE)
cat(sprintf("   rho = %.4f\n", result$estimate))
cat(sprintf("   p = %.6e\n", result$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 6. Test de Kendall
# ----------------------------------------------------------------------------
cat("6. Test de corrélation de Kendall\n")
result <- cor.test(x, y, method = "kendall", exact = FALSE)
cat(sprintf("   tau = %.4f\n", result$estimate))
cat(sprintf("   p = %.6e\n", result$p.value))
cat("\n")

cat("=== Benchmarks R terminés ===\n")
