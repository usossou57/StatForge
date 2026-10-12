
# ============================================================================
# Benchmarks R pour validation croisée (Phase 17 — Inférence V2)
# ============================================================================
# Ce script calcule les mêmes statistiques que les fonctions Python
# pour validation croisée. Les valeurs attendues sont utilisées dans
# tests/test_statistics/test_tests_parametric.py (TestRBenchmarkValidation).
#
# Exécution : Rscript benchmarks/r/phase17_benchmarks.R
# ============================================================================

# Installation des packages si nécessaire
if (!require("stats")) install.packages("stats")

cat("=== Benchmarks R pour Phase 17 ===\n\n")

# ----------------------------------------------------------------------------
# 1. Test t pairé
# ----------------------------------------------------------------------------
cat("1. Test t pairé\n")
before <- c(10.2, 11.5, 9.8, 10.9, 11.2, 10.5, 11.8, 10.1, 10.7, 11.0)
after <- c(11.5, 12.1, 10.5, 11.8, 12.0, 11.2, 12.5, 10.9, 11.4, 11.7)

result <- t.test(before, after, paired = TRUE)
cat(sprintf("   t = %.4f\n", result$statistic))
cat(sprintf("   df = %.1f\n", result$parameter))
cat(sprintf("   p = %.6f\n", result$p.value))
cat(sprintf("   IC 95%% = [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))
cat(sprintf("   Moyenne différence = %.4f\n", result$estimate))
cat("\n")

# ----------------------------------------------------------------------------
# 2. Test t de Welch
# ----------------------------------------------------------------------------
cat("2. Test t de Welch\n")
group_a <- c(10.2, 11.5, 9.8, 10.9, 11.2)
group_b <- c(12.5, 13.1, 11.8, 12.9, 13.5)
df <- data.frame(
  outcome = c(group_a, group_b),
  group = factor(c(rep("A", 5), rep("B", 5)))
)

result <- t.test(outcome ~ group, data = df, var.equal = FALSE)
cat(sprintf("   t = %.4f\n", result$statistic))
cat(sprintf("   df = %.1f\n", result$parameter))
cat(sprintf("   p = %.6f\n", result$p.value))
cat(sprintf("   IC 95%% = [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))
cat(sprintf("   Moyenne A = %.4f\n", result$estimate[1]))
cat(sprintf("   Moyenne B = %.4f\n", result$estimate[2]))
cat("\n")

# ----------------------------------------------------------------------------
# 3. Test t à un échantillon
# ----------------------------------------------------------------------------
cat("3. Test t à un échantillon\n")
values <- c(5.2, 6.1, 5.8, 5.5, 6.0, 5.9, 5.7, 6.2, 5.6, 5.8)

result <- t.test(values, mu = 5.0)
cat(sprintf("   t = %.4f\n", result$statistic))
cat(sprintf("   df = %.1f\n", result$parameter))
cat(sprintf("   p = %.6f\n", result$p.value))
cat(sprintf("   IC 95%% = [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))
cat(sprintf("   Moyenne = %.4f\n", result$estimate))
cat("\n")

# ----------------------------------------------------------------------------
# 4. Test de proportion (z)
# ----------------------------------------------------------------------------
cat("4. Test de proportion (z)\n")
successes <- 60
n_total <- 100
p0 <- 0.5

result <- prop.test(successes, n_total, p = p0, correct = FALSE)
cat(sprintf("   Chi² = %.4f\n", result$statistic))
cat(sprintf("   p = %.6f\n", result$p.value))
cat(sprintf("   Proportion = %.4f\n", result$estimate))
cat(sprintf("   IC 95%% = [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))
cat("\n")

# Test exact binomial
result_exact <- binom.test(successes, n_total, p = p0)
cat(sprintf("   Test exact : p = %.6f\n", result_exact$p.value))
cat("\n")

# ----------------------------------------------------------------------------
# 5. Test exact de Fisher
# ----------------------------------------------------------------------------
cat("5. Test exact de Fisher\n")
# Tableau de contingence 2x2
#       X   Y
#   A   8   2
#   B   3   7
contingency <- matrix(c(8, 3, 2, 7), nrow = 2, byrow = FALSE)
rownames(contingency) <- c("A", "B")
colnames(contingency) <- c("X", "Y")

result <- fisher.test(contingency)
cat(sprintf("   Odds Ratio = %.4f\n", result$estimate))
cat(sprintf("   p = %.6f\n", result$p.value))
cat(sprintf("   IC 95%% = [%.4f, %.4f]\n", result$conf.int[1], result$conf.int[2]))
cat("\n")

# ----------------------------------------------------------------------------
# 6. Test de variances (Levene via car package)
# ----------------------------------------------------------------------------
cat("6. Test de variances\n")
if (!require("car")) {
  cat("   Package 'car' non installé. Installation...\n")
  install.packages("car", quiet = TRUE)
  library("car")
}

# Levene test
result_levene <- leveneTest(outcome ~ group, data = df, center = "median")
cat(sprintf("   Levene F = %.4f\n", result_levene[1, "F value"]))
cat(sprintf("   Levene p = %.6f\n", result_levene[1, "Pr(>F)"]))

# Bartlett test
result_bartlett <- bartlett.test(outcome ~ group, data = df)
cat(sprintf("   Bartlett Chi² = %.4f\n", result_bartlett$statistic))
cat(sprintf("   Bartlett p = %.6f\n", result_bartlett$p.value))
cat("\n")

cat("=== Benchmarks R terminés ===\n")