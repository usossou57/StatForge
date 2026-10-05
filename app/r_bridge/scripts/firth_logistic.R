#!/usr/bin/env Rscript

library(jsonlite)
library(logistf)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript firth_logistic.R input.json output.json")
}

input_file <- args[1]
output_file <- args[2]

tryCatch({
  input_data <- fromJSON(input_file)
  data <- as.data.frame(input_data$data)
  outcome <- input_data$outcome
  predictors <- input_data$predictors

  formula_str <- paste(outcome, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)

  model <- logistf(formula_obj, data = data)

  # 1. Coefficients (vecteur nommé, attribut direct)
  coef_vec <- model$coefficients
  coef_names <- names(coef_vec)

  # 2. Erreurs standards (diagonale de la matrice de variance-covariance)
  se_vec <- sqrt(diag(model$var))
  names(se_vec) <- coef_names

  # 3. P-values calculées manuellement (Wald test : z = coef/se, p = 2*pnorm(-|z|))
  z_vec <- coef_vec / se_vec
  p_vec <- 2 * pnorm(-abs(z_vec))
  names(p_vec) <- coef_names

  # 4. Intervalles de confiance (attributs directs du modèle, vecteurs simples)
  ci_lower_vec <- model$ci.lower
  ci_upper_vec <- model$ci.upper

  # Construction des listes nommées pour JSON
  coefficients <- as.list(coef_vec)
  standard_errors <- as.list(se_vec)
  p_values <- as.list(p_vec)

  confidence_intervals <- list()
  for (nm in coef_names) {
    confidence_intervals[[nm]] <- list(ci_lower_vec[nm], ci_upper_vec[nm])
  }

  results <- list(
    coefficients = coefficients,
    standard_errors = standard_errors,
    p_values = p_values,
    confidence_intervals = confidence_intervals,
    n_obs = as.integer(model$n),
    log_likelihood = as.numeric(model$loglik[2]),
    converged = TRUE,
    warnings = character(0)
  )

  write_json(results, output_file, auto_unbox = TRUE, pretty = TRUE)

}, error = function(e) {
  error_result <- list(
    error = conditionMessage(e),
    converged = FALSE
  )
  write_json(error_result, output_file, auto_unbox = TRUE, pretty = TRUE)
})
