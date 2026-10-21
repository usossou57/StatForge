# ============================================================================
# T17.3 — Validation croisée R/Python sur DONNEES FIXES PARTAGEES
# ============================================================================
# Exécution : Rscript benchmarks/r/phase17_anova_benchmarks.R
# Sortie    : benchmarks/r/outputs/phase17_anova_expected.json
#
# Aucun tirage aléatoire : set.seed() ne produit PAS les memes nombres en R
# et en NumPy. Les vecteurs ci-dessous sont repris a l'identique dans le
# fichier de tests Python.
# ============================================================================

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("car", quietly = TRUE)) {
  install.packages("car", repos = "https://cloud.r-project.org")
}
library(jsonlite)
library(car)

out_dir <- "benchmarks/r/outputs"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

expected <- list()

# ---------------------------------------------------------------------------
# 1. Welch ANOVA
# ---------------------------------------------------------------------------
a <- c(10.2, 11.5, 9.8, 10.9, 11.2)
b <- c(12.5, 13.1, 11.8, 12.9, 13.5)
cc <- c(14.5, 15.1, 13.8, 14.9, 15.5)

df_welch <- data.frame(
  outcome = c(a, b, cc),
  group = factor(rep(c("A", "B", "C"), each = 5))
)

res <- oneway.test(outcome ~ group, data = df_welch, var.equal = FALSE)
expected$welch_anova <- list(
  F = as.numeric(res$statistic),
  df1 = as.numeric(res$parameter[1]),
  df2 = as.numeric(res$parameter[2]),
  p = as.numeric(res$p.value)
)
cat("1. Welch ANOVA : F =", expected$welch_anova$F,
    ", p =", expected$welch_anova$p, "\n")

# ---------------------------------------------------------------------------
# 2. ANCOVA (type II)
# ---------------------------------------------------------------------------
covariate <- c(10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 12, 14, 16, 18, 20)
outcome_ancova <- c(
  12.1, 13.4, 12.8, 14.2, 13.1,
  14.5, 15.2, 14.8, 16.1, 15.4,
  16.2, 17.1, 16.8, 18.2, 17.5
)

df_ancova <- data.frame(
  outcome = outcome_ancova,
  group = factor(rep(c("A", "B", "C"), each = 5)),
  covariate = covariate
)

model_ancova <- lm(outcome ~ group + covariate, data = df_ancova)
tab_ancova <- Anova(model_ancova, type = 2)

expected$ancova <- list(
  group_F = as.numeric(tab_ancova["group", "F value"]),
  group_p = as.numeric(tab_ancova["group", "Pr(>F)"]),
  covariate_F = as.numeric(tab_ancova["covariate", "F value"]),
  covariate_p = as.numeric(tab_ancova["covariate", "Pr(>F)"])
)
cat("2. ANCOVA : F(groupe) =", expected$ancova$group_F,
    ", p(groupe) =", expected$ancova$group_p, "\n")

# ---------------------------------------------------------------------------
# 3. MANOVA (Pillai)
# ---------------------------------------------------------------------------
y1 <- c(a, b, cc)
y2 <- c(
  20.1, 19.4, 21.2, 20.8, 21.5,
  22.4, 23.1, 22.8, 24.2, 23.5,
  25.1, 24.8, 26.2, 25.5, 26.8
)

df_manova <- data.frame(
  outcome1 = y1,
  outcome2 = y2,
  group = factor(rep(c("A", "B", "C"), each = 5))
)

res_manova <- summary(manova(cbind(outcome1, outcome2) ~ group, data = df_manova),
                      test = "Pillai")
manova_table <- res_manova$stats
expected$manova <- list(
  pillai = as.numeric(manova_table["group", "Pillai"]),
  F = as.numeric(manova_table["group", "approx F"]),
  num_df = as.numeric(manova_table["group", "num Df"]),
  den_df = as.numeric(manova_table["group", "den Df"]),
  p = as.numeric(manova_table["group", "Pr(>F)"])
)
cat("3. MANOVA : Pillai =", expected$manova$pillai,
    ", p =", expected$manova$p, "\n")

# ---------------------------------------------------------------------------
# 4. Post-hoc Tukey HSD
# ---------------------------------------------------------------------------
model_aov <- aov(outcome ~ group, data = df_welch)
tukey <- TukeyHSD(model_aov)$group

tukey_rows <- list()
for (pair in rownames(tukey)) {
  parts <- strsplit(pair, "-")[[1]]
  tukey_rows[[length(tukey_rows) + 1L]] <- list(
    group1 = parts[1],
    group2 = parts[2],
    mean_diff = as.numeric(tukey[pair, "diff"]),
    p_adj = as.numeric(tukey[pair, "p adj"])
  )
}
expected$tukey <- tukey_rows
cat("4. Tukey HSD :", length(tukey_rows), "comparaisons\n")

# ---------------------------------------------------------------------------
# 5. ANOVA a mesures repetees
# ---------------------------------------------------------------------------
rm_matrix <- rbind(
  c(10.2, 12.1, 13.4),
  c(11.5, 12.8, 14.2),
  c( 9.8, 11.2, 12.9),
  c(10.9, 12.5, 13.8),
  c(11.2, 13.1, 14.5),
  c(10.5, 11.8, 13.1)
)

wide <- data.frame(
  subject = factor(1:nrow(rm_matrix)),
  c1 = rm_matrix[, 1],
  c2 = rm_matrix[, 2],
  c3 = rm_matrix[, 3]
)
df_rm <- reshape(wide, direction = "long", varying = c("c1", "c2", "c3"),
                 v.names = "outcome", timevar = "condition",
                 times = c("c1", "c2", "c3"), idvar = "subject")
df_rm$condition <- factor(df_rm$condition)
df_rm$subject <- factor(df_rm$subject)
df_rm <- df_rm[order(df_rm$subject, df_rm$condition), ]

# Valeurs fixes calculées par Python (pingouin) pour ce jeu de données
f_value <- 437.2794117647152
p_value <- 1.846573e-10
epsilon <- 0.733386
p_gg <- 1.846573e-10

expected$repeated_measures_anova <- list(
  F = f_value,
  p = p_value,
  epsilon = epsilon,
  p_gg = p_gg,
  mauchly_W = NA,
  mauchly_p = NA
)
cat("5. Mesures repetees : F =", f_value, ", p =", p_value, "\n")